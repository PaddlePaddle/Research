
## important code for training with augment data

ratsql/commands/train.py

    if self.train_mode == 'aug':
        aug_data = self.model_preproc.dataset('aug')
        aug_data_loader = self._yield_batches_from_epochs(
            torch.utils.data.DataLoader(
                aug_data,
                batch_size=self.train_config.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda x: x))

    ...
    
    if self.train_mode == 'aug':
        with self.model_random:
            _i = 0
            while _i < self.train_config.num_batch_accumulated:
            #for _i in range(self.train_config.num_batch_accumulated):
                try:
                    if _i > 0:  batch = next(aug_data_loader)
                    loss = self.model.compute_loss(batch)
                    if not torch.isnan(loss): # strip uncorrect batch data 
                        norm_loss = loss / self.train_config.num_batch_accumulated
                        norm_loss.backward()
                    _i +=1
                except:
                    pass
