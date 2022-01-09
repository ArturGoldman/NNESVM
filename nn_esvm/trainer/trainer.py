import torch
from tqdm import tqdm

from nn_esvm.base import BaseTrainer
from nn_esvm.utils import inf_loop, MetricTracker
from nn_esvm.datasets.utils import get_dataloader
from nn_esvm.MCMC import GenMCMC
import nn_esvm.distributions
import nn_esvm.functions_to_E
import nn_esvm.loss
import nn_esvm.datasets
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from multiprocessing import Pool
import multiprocessing
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            optimizer,
            config,
            device,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True
    ):
        super().__init__(model, optimizer, lr_scheduler, config, device)
        self.skip_oom = skip_oom
        self.config = config

        self.criterion = config.init_obj(config["loss_spec"], nn_esvm.loss)
        self.function = config.init_ftn("f(x)", nn_esvm.functions_to_E)
        self.target_distribution = config.init_obj(config["data"]["target_dist"],
                                                   nn_esvm.distributions)
        self.data_loader = get_dataloader(config, self.target_distribution, "train")

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(self.data_loader)
            self.len_epoch = len_epoch

        self.log_step = self.config["trainer"]["log_step"]
        self.val_step = self.config["trainer"]["val_step"]
        self.cv_type = self.config["cv_type"]

        self.train_metrics = MetricTracker(
            "loss_esv", "grad_norm",
            writer=self.writer
        )

        self.trial_num = self.config["data"]["val"]["Trials"]
        self.baseline_box = None

        self.val_desc = self.config["data"]["val"]["datasets"][0]["args"]
        self.generator = GenMCMC(self.target_distribution.grad_log,
                                 self.val_desc["mcmc_type"], self.val_desc["gamma"])

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                loss = self.process_batch(
                    batch,
                    metrics=self.train_metrics
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad_norm", self.get_grad_norm(self.model))
            if batch_idx % self.log_step == 0 or (batch_idx + 1 == self.len_epoch and epoch == self.epochs):
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss (ESV): {:.6f}".format(
                        epoch, self._progress(batch_idx), loss
                    )
                )

                self._log_scalars(self.train_metrics)
            if batch_idx + 1 >= self.len_epoch:
                break

        self.lr_scheduler.step()
        self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
        if epoch % self.val_step == 0 or epoch == self.epochs:
            self._valid_example()

        log = self.train_metrics.result()

        return log

    def process_cv(self, batch, cr_gr=False, mode="simple_additive"):
        if mode == "simple_additive":
            return self.model(batch)
        elif mode == "stein":
            laplacians = []
            grads = []
            for x in tqdm(batch):
                out = self.model(x)
                grad = torch.autograd.grad(out, x, create_graph=cr_gr)
                grads.append(grad[0])

                hess = torch.autograd.functional.hessian(self.model, x, create_graph=cr_gr)
                laplacians.append(torch.trace(hess))
            log_grads = self.target_distribution.grad_log(batch)
            return torch.stack(laplacians) + (log_grads*torch.stack(grads, dim=0)).sum(dim=1)
        else:
            raise ValueError("Unrecognised CV type")

    def process_batch(self, batch: torch.Tensor, metrics: MetricTracker):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        batch.requires_grad = True
        outputs = self.process_cv(batch, cr_gr=True, mode=self.cv_type)

        loss_esv = self.criterion(self.function(batch)-outputs)
        loss_esv.backward()
        self._clip_grad_norm()
        self.optimizer.step()

        metrics.update("loss_esv", loss_esv.item())

        return loss_esv.item()

    def generate_parallel_chains(self, T):
        rseed = 926
        nbcores = multiprocessing.cpu_count()
        print("Total cores for multiprocessing", nbcores)
        multi = Pool(nbcores)
        #starting_points = torch.randn((T, self.target_distribution.dim))
        res = multi.starmap(self.generator.gen_samples,
                            [(self.val_desc["n_burn"]+self.val_desc["n_clean"],
                              self.target_distribution.dim,
                              rseed + i) for i in range(T)])
        return res

    def calc_box(self, description, to_cv=False):
        # TODO: mean estimation is unstable
        cur_box = []
        vnfs = []
        vnfcvs = []
        print("Parallel chain generation started")
        st_time = time.time()
        chains = self.generate_parallel_chains(self.trial_num)
        fin_time = time.time()
        print(self.trial_num, "chains generated in", fin_time-st_time, "seconds")
        chains = torch.stack(chains, dim=0)
        chains = chains[:, self.val_desc["n_burn"]:, :]

        for T in tqdm(range(self.trial_num), desc=description):
            batch = chains[T].to(self.device)
            batch.requires_grad = True
            if to_cv:
                vnf = self.criterion(self.function(chains[T]))
                cvs = self.process_cv(batch, False, self.cv_type)
                vnfcv = self.criterion(self.function(batch) - cvs)
                cur_box.append((self.function(batch)-cvs).mean().item())

                vnfs.append(vnf.item())
                vnfcvs.append(vnfcv.item())

            else:
                cur_box.append(self.function(batch).mean().item())

        if len(vnfs) == 0:
            return cur_box, None, None
        return cur_box, sum(vnfs) / len(vnfs), sum(vnfcvs) / len(vnfcvs)

    def _valid_example(self):
        """
        update statistics on newly generated chains
        """
        self.model.eval()

        if self.baseline_box is None:
            self.baseline_box, _, _ = self.calc_box("Calculating baseline")

        cur_box, vnf, vnfcv = self.calc_box("Validating", to_cv=True)

        self.writer.add_scalar("V_n(f)", vnf)
        self.writer.add_scalar("V_n(f-g)", vnfcv)
        self.writer.add_scalar("V_n(f) over V_n(f-g)", vnf/vnfcv)

        self._log_boxplots([self.baseline_box, cur_box], ["Vanila", "ESVM"])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_boxplots(self, boxes, names):
        plt.figure(figsize=(12, 8))
        plt.boxplot(boxes, showfliers=False, labels=names)
        plt.title("BananaShape")
        plt.grid()
        self.writer.add_image("Boxplots", plt)
        self.writer.add_plot("Boxplots_plotty", plt)

    @staticmethod
    @torch.no_grad()
    def get_grad_norm(model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
