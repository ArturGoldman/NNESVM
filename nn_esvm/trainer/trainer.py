import torch
from tqdm import tqdm

from nn_esvm.base import BaseTrainer
from nn_esvm.utils import inf_loop, MetricTracker
from nn_esvm.datasets.utils import get_dataloader
from nn_esvm.MCMC import GenMCMC
from nn_esvm.CV import process_cv
import nn_esvm.distributions
import nn_esvm.functions_to_E
import nn_esvm.loss
import nn_esvm.datasets
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
import multiprocessing
import time
import pandas as pd


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

        self.logger.info("Run_id: {}".format(config.run_id))

        self.skip_oom = skip_oom
        self.config = config

        self.criterion = config.init_obj(config["loss_spec"], nn_esvm.loss)
        self.metric = config.init_obj(config["metric"], nn_esvm.loss)
        self.function = config.init_ftn("f(x)", nn_esvm.functions_to_E)
        self.target_distribution = config.init_obj(config["data"]["target_dist"],
                                                   nn_esvm.distributions)
        self.data_loader = get_dataloader(config, self.target_distribution, "train", self.writer)

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
        self.out_model_dim = self.config["arch"]["args"]["out_dim"]

        self.train_metrics = MetricTracker(
            "loss_esv", "grad_norm",
            writer=self.writer
        )

        self.trial_num = self.config["data"]["val"]["Trials"]
        self.baseline_box = None

        self.val_desc = self.config["data"]["val"]["datasets"][0]["args"]
        self.val_chains = None
        if self.config["data"]["val"].get("folder_name", None) is not None:
            checkpoint = torch.load(self.config["data"]["val"]["folder_name"])
            chains = checkpoint["chains"]
            chains = chains[:, ::self.val_desc["n_step"], :]
            self.val_chains = chains[1:]

        self.val_generator = GenMCMC(self.target_distribution,
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

    def process_batch(self, batch: torch.Tensor, metrics: MetricTracker):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        outputs = process_cv(self.model, batch, self.device, self.out_model_dim,
                             self.target_distribution.dim, self.target_distribution.grad_log,
                             cr_gr=True, mode=self.cv_type)
        loss_esv = self.criterion(self.function(batch), outputs)
        # smart loss does backward itself
        if "Smart" not in self.config["loss_spec"]["type"]:
            loss_esv.backward()
        self._clip_grad_norm()
        self.optimizer.step()
        metrics.update("loss_esv", loss_esv.item())

        return loss_esv.item()

    @staticmethod
    def box_only(func, chain):
        return func(chain)

    def calc_box(self, description, to_cv=False):
        """
        for T in tqdm(range(self.trial_num), desc=description):
            batch = chains[T].to(self.device)
            batch.requires_grad = True
            if to_cv:
                vnf = self.metric(self.function(chains[T]))
                cvs = self.process_cv(batch, False, self.cv_type)
                vnfcv = self.metric(self.function(batch) - cvs)
                cur_box.append((self.function(batch)-cvs).mean().item())

                vnfs.append(vnf.item())
                vnfcvs.append(vnfcv.item())

            else:
                cur_box.append(self.function(batch).mean().item())

        if len(vnfs) == 0:
            return cur_box, None, None
        return cur_box, sum(vnfs) / len(vnfs), sum(vnfcvs) / len(vnfcvs)
        """

        if self.val_chains is None:
            print("~~~~~~", description, "~~~~~~")
            print("Parallel chain generation started")
            st_time = time.time()
            chains = self.val_generator.generate_parallel_chains(self.val_desc["n_burn"], self.val_desc["n_clean"],
                                                                 self.trial_num,
                                                                 self.val_desc["rseed"])
            fin_time = time.time()
            print(self.trial_num, "chains generated in", fin_time-st_time, "seconds")
            chains = torch.stack(chains, dim=0)
            chains = chains[:, ::self.val_desc["n_step"], :]
            self.val_chains = chains

        nbcores = multiprocessing.cpu_count()
        ctx = torch.multiprocessing.get_context('spawn')
        print("Total cores for multiprocessing", nbcores)

        print("Parallel f(chain) calculation started")
        with ctx.Pool(nbcores) as multi:
            f_chains = multi.starmap(self.box_only,
                                     [(self.function, self.val_chains[i]) for i in range(self.trial_num)])
        f_chains = torch.stack(f_chains, dim=0)
        print("Parallel f(chain) calculation finished")
        if to_cv:
            # Calculate Empirical Spectral Variance
            print("Parallel vnfs calculation started")
            with ctx.Pool(nbcores) as multi:
                vnfs = multi.starmap(self.metric,
                             [(f_chains[i], None) for i in range(self.trial_num)])
            vnfs = torch.stack(vnfs, dim=0)
            print("Parallel cvs calculation started")
            self.model = self.model.to('cpu')
            with ctx.Pool(nbcores) as multi:
                cvs = multi.starmap(process_cv,
                                [(self.model, self.val_chains[i], 'cpu',
                                  self.out_model_dim, self.target_distribution.dim,
                                  self.target_distribution.grad_log,
                                  False, self.cv_type, True) for i in range(self.trial_num)])
            self.model = self.model.to(self.device)
            cvs = torch.stack(cvs, dim=0)
            print("Parallel vnfcvs calculation started")
            with ctx.Pool(nbcores) as multi:
                vnfcvs = multi.starmap(self.metric,
                               [(f_chains[i], cvs[i]) for i in range(self.trial_num)])
            vnfcvs = torch.stack(vnfcvs, dim=0)
            print("Evaluations finished")

            # calculate Empirical Variance
            v_base = torch.var(f_chains, dim=(-1, -2), unbiased=False).mean()
            v_cur = torch.var(f_chains-cvs, dim=(-1, -2), unbiased=False).mean()
            return (f_chains-cvs).mean(dim=(-1, -2)), vnfs.mean(), vnfcvs.mean(), v_base, v_cur
        else:
            return f_chains.mean(dim=(-1, -2)), None, None, None, None

    def _valid_example(self):
        """
        update statistics on newly generated chains
        """
        self.model.eval()

        if self.baseline_box is None:
            self.baseline_box, _, _, _, _ = self.calc_box("Calculating baseline")

        cur_box, vnf, vnfcv, v_base, v_cur = self.calc_box("Validating", to_cv=True)

        self.writer.add_scalar("V_n(f), ESV", vnf)
        self.writer.add_scalar("V_n(f-g), ESV", vnfcv)
        self.writer.add_scalar("V_n(f) over V_n(f-g), ESV", vnf/vnfcv)
        self.writer.add_scalar("V_n(f), EV", v_base)
        self.writer.add_scalar("V_n(f-g), EV", v_cur)
        self.writer.add_scalar("V_n(f) over V_n(f-g), EV", v_base/v_cur)
        base_v = torch.var(self.baseline_box)
        base_c = torch.var(cur_box)
        self.writer.add_text("Var of means in boxplot (including outliers)",
                             "Baseline var: {}, current var: {}, current/baseline:{}".format(base_v, base_c, base_v/base_c))
        srt_b, _ = torch.sort(self.baseline_box)
        srt_c, _ = torch.sort(cur_box)
        df = pd.DataFrame({"f": srt_b.tolist(),
                           "f-g": srt_c.tolist()})
        self.writer.add_table("num_boxes", df)

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
