import time
import pickle 
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, util
import wandb

if __name__ == '__main__':
    opt = TrainOptions().parse()
    wandb.init(project = 'NeurImg-HDR', config = opt, settings=wandb.Settings(start_method='fork'))
    wandb.run.name = opt.name
    opt = wandb.config
    
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt) 
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data) 
            model.optimize_parameters() 

            if total_iters % opt.display_freq == 0:  
                # -------- Save Training images ----------
                model.compute_visuals()
                if opt.model == 'image' or opt.model == 'lfn':
                    visualizer.save_image_to_disk(model.get_current_visuals(), epoch_iter, epoch)
                else:
                    visualizer.newvideo_save_image(model.get_current_visuals(), epoch_iter, epoch)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        print('End of epoch %d / %d \t Time Taken: %d sec \t name: %s' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, opt.name))
        model.update_learning_rate()
    wandb.finish()
