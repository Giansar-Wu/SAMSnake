import time
import datetime
import torch
import tqdm

from..recorder.recorder import Recorder

class Trainer(object):
    def __init__(self, network: torch.nn.Module):
        network = network.cuda()
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    # def train(self, epoch, data_loader, optimizer, recorder):
    #     max_iter = len(data_loader)
    #     self.network.train()
    #     end = time.time()
    #     for iteration, batch in enumerate(data_loader):
    #         data_time = time.time() - end
    #         iteration = iteration + 1
    #         recorder.step += 1

    #         batch = self.to_cuda(batch)
    #         batch.update({'epoch': epoch})
    #         output, loss, loss_stats = self.network(batch)
            
    #         loss = loss.mean()
    #         optimizer.zero_grad()
            
    #         loss.backward()
    #         torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
    #         optimizer.step()

    #         loss_stats = self.reduce_loss_stats(loss_stats)
    #         recorder.update_loss_stats(loss_stats)

    #         batch_time = time.time() - end
    #         end = time.time()
    #         recorder.batch_time.update(batch_time)
    #         recorder.data_time.update(data_time)

    #         if iteration % 20 == 0 or iteration == (max_iter - 1):
    #             eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
    #             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #             lr = optimizer.param_groups[0]['lr']
    #             memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

    #             training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
    #             training_state = training_state.format(eta_string, str(recorder), lr, memory)
    #             print(training_state)

    #             recorder.record('train')


    def train(self, epoch: int, max_epoch: int, data_loader, optimizer, recorder: Recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            batch.update({'epoch': epoch})

            output, loss, loss_stats = self.network(batch)

            loss = loss.mean()
            optimizer.zero_grad()
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 50 == 0 or iteration == (max_iter - 1):
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                eta_end_seconds = recorder.batch_time.global_avg * (max_iter - iteration + max_iter * (max_epoch - epoch - 1))
                eta_end_string = str(datetime.timedelta(seconds=int(eta_end_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                # training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                # training_state = training_state.format(eta_string, str(recorder), lr, memory)
                training_state = F"{str(recorder)} lr:{lr:.6f} max_mem:{memory:.0f}MiB epoch_eta:{eta_string} eta:{eta_end_string}"
                recorder.logger.info(training_state)
                # print(training_state)

                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder: Recorder=None):
        self.network.eval()
        # torch.cuda.empty_cache()
        val_loss_stats = {}
        # for batch in tqdm.tqdm(data_loader):
        for batch in data_loader:
            batch = self.to_cuda(batch)
            batch.update({'epoch': epoch})
            with torch.no_grad():
                output = self.network(batch)
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

        if evaluator is not None:
            result, summarize_info = evaluator.summarize()
            if summarize_info:
                for info_msg in summarize_info:
                    recorder.logger.info(info_msg)
            val_loss_stats.update({"ap":result})

        if recorder:
            recorder.record('val', epoch, val_loss_stats)

    def get_reduced(self, tensor, current_device, dest_device, world_size):
        tensor = tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
        tensor = tensor.to(current_device)
        torch.distributed.reduce(tensor, dst=dest_device)
        tensor_mean = tensor.item() / world_size
        return tensor_mean
    
    def train_ddp(self, model:torch.nn.Module, epoch: int, max_epoch: int, data_loader, optimizer, recorder: Recorder, local_rank: int, world_size: int):
        max_iter = len(data_loader)
        model.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            batch = self.to_cuda(batch)
            batch.update({'epoch': epoch})

            output, loss, loss_stats = model(batch)

            loss = loss.mean()
            optimizer.zero_grad()
            
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 40)
            optimizer.step()

            # loss_stats = self.reduce_loss_stats(loss_stats)
            # recorder.update_loss_stats(loss_stats)
            loss_stats = self.reduce_loss_stats(loss_stats)
            loss_stats =  {k: self.get_reduced(v, local_rank, 0, world_size) for k, v in loss_stats.items()}
            recorder.update_loss_stats_ddp(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)
            if local_rank == 0 and (iteration % 50 == 0 or iteration == (max_iter - 1)):
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                eta_end_seconds = recorder.batch_time.global_avg * (max_iter - iteration + max_iter * (max_epoch - epoch - 1))
                eta_end_string = str(datetime.timedelta(seconds=int(eta_end_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                # training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                # training_state = training_state.format(eta_string, str(recorder), lr, memory)
                training_state = F"{str(recorder)} lr:{lr:.6f} max_mem:{memory:.0f}MiB epoch_eta:{eta_string} eta:{eta_end_string}"
                recorder.logger.info(training_state)
                # print(training_state)

                recorder.record('train')
    
    def val_ddp(self, model, epoch, data_loader, evaluator, recorder: Recorder, local_rank: int, world_size: int):
        model.eval()
        # torch.cuda.empty_cache()
        val_loss_stats = {}
        # for batch in tqdm.tqdm(data_loader):
        with torch.no_grad():
            for batch in data_loader:
                batch = self.to_cuda(batch)
                batch.update({'epoch': epoch})

                output = model(batch)

                if evaluator is not None:
                    evaluator.evaluate(output, batch)

        if (local_rank == 0) and (evaluator is not None):
            result, summarize_info = evaluator.summarize()
            if summarize_info:
                for info_msg in summarize_info:
                    recorder.logger.info(info_msg)
            val_loss_stats.update({"ap":result})

        if local_rank == 0 and recorder:
            recorder.record('val', epoch, val_loss_stats)
