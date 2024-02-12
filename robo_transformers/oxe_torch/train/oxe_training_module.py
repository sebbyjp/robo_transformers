
from torch import optim, nn
import lightning as L
from einops import rearrange, repeat
from oxe_torch.action_tokenization import RTX1ActionTokenizer as ActionTokenizer
import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import CombinedLoader

from absl import flags, app
from gym import spaces
from collections import OrderedDict
import tensorflow_hub as hub
import tensorflow as tf
import wandb
import numpy as np
from oxe_torch.data.dataset import TorchRLDSDataset, get_oxe_dataset
from oxe_torch.rt1.maruya24_rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from oxe_torch.rt1.maruya24_rt1.transformer_network import TransformerNetwork
from oxe_torch.rt1.maruya24_rt1.transformer_network_test_set_up import state_space_list
import copy
from pytorch_lightning.loggers import WandbLogger
import h5py
from torchvision import transforms
from importlib_resources import files
FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_integer("num_warmup_steps", 1000, "Number of warmup steps.")
flags.DEFINE_integer("shuffle_buffer_size", 1000, "Shuffle buffer size.")
flags.DEFINE_integer("eval_batch_size", 1, "Eval Batch size.")
flags.DEFINE_float("lr", 1e-3, "Learning Rate.")
flags.DEFINE_float("min_lr", 1e-6, "Min Learning Rate.")
flags.DEFINE_float("weight_decay", 0, "Weight Decay.")
flags.DEFINE_string("dataset_name", "fractal20220817_data", "Dataset name.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory.")
flags.DEFINE_list("baselines", [], "Baselines to evaluate against.")
flags.DEFINE_bool("data_augmentation", True,
                  "Whether or not to use data augmentation.")
flags.DEFINE_float("conditioning_scale", 1.0,
                   "Scale of film conditioning. on text input.")
flags.DEFINE_float("label_smoothing", 0.0, "Label smoothing.")
flags.DEFINE_string("loss", "cse", "Loss function.")
flags.DEFINE_bool("freeze_vit", False, "Freeze ViT weights.")
flags.DEFINE_string("weights_path", None, "Path to weights to load.")
tf.config.set_visible_devices([], "GPU")

# set lightning device to cpu if no gpu is available
def dict_to_device(dict_obj, device, dtype=None):
    """
    put all the values in the [dict_obj] to [device]
    """
    for _, v in dict_obj.items():
        v = torch.tensor(v, device=device, dtype=dtype)
    return dict_obj


def retrieve_single_timestep(dict_obj, idx):
    """
    get all the values in the [dict_obj] at index [idx]
    v[:, idx], all the values in the dictionary at second dimension needs to be same
    """
    dict_obj_return = copy.deepcopy(dict_obj)
    for k, v in dict_obj.items():
        dict_obj_return[k] = v[:, idx]
    return dict_obj_return

class LazyTFModule:
    """Lazy loads a tensorflow module."""

    def __init__(self, url: str):
        self.url = url
        self.module = None

    def __getattr__(self, name: str):
        if self.module is None:
            self.module = hub.load(self.url)
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        if self.module is None:
            self.module = hub.load(self.url)
        return self.module(*args, **kwargs)


TEXT_ENCODER = LazyTFModule(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)

def embed_text(input: list[str] | str, batch_size: int = 1) -> tf.Tensor:
    """Embeds a string using the Universal Sentence Encoder. Copies the string
        to fill the batch dimension.

    Args:
        input (str): The string to embed.
        batch_size (int, optional): . Defaults to 1.

    Returns:
        tf.Tensor: A tensor of shape (batch_size, 512).
    """
    with torch.no_grad():
        if isinstance(input, str):
            input = input.lstrip(' ').rstrip(' ')
            input = np.tile(np.array(input), (batch_size,))
    return np.ascontiguousarray(TEXT_ENCODER(input))

CHECKPOINT_CALLBACK = ModelCheckpoint(
    monitor="accuracy",
    mode="max",
    filename="rtx-custom-{epoch:02d}-{accuracy:.2f}",
    dirpath="checkpoints/rtx_custom",
    every_n_train_steps=100
)

class LogCallback(L.Callback):
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        return super().on_train_epoch_end(trainer, pl_module)


class HD5Dataset(IterableDataset):
    def __init__(self, hdf5_file, transform=None, window_size=6, future_action_window_size=0, shuffle=False, seed=42):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file =  h5py.File(hdf5_file, 'r')
        self.transform = transform
        if self.transform is None:
            # Define your transformations / augmentations
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        self.window_size = window_size
        # self.future_action_window_size = future_action_window_size
        self.shuffle = shuffle
        self.seed = seed
             # Open the hdf5 file and get the size of the dataset
        self.length =self.file.attrs['size']
        self.idxs = np.arange(0, self.length)

        self.text_encoder = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        # self.instructions = []
        # for i in range(self.length):
        #     instruction = torch.zeros((1,512), dtype=torch.float32, device='cpu')
        #     instruction[:,:] = self.text_encoder([self.file['observation/language_instruction'][i][0].decode()]).numpy()
        #     self.instructions.append(instruction)
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.idxs)

    def __len__(self):
        return self.length

    def __iter__(self):
        for idx in self.idxs:
            observation = torch.zeros((self.window_size, 224, 224, 3))
            # image_wrists = torch.zeros((self.window_size, 128, 128, 3))
            action = torch.zeros((self.window_size, 7))
            if idx < self.window_size - 1:
                observation[self.window_size-idx - 1:,:,:,:] = torch.stack([self.transform(self.file['observation/image_head'][i]) for i in range(0, idx+1)]).permute(0, 2, 3, 1)
                # image_wrists[self.window_size-idx - 1:,:,:,:] = torch.stack([self.transform(self.file['observation/image_wrist_left'][i]) for i in range(0, idx+1)]).permute(0, 2, 3, 1)
                action[self.window_size-idx - 1:,:] = torch.tensor([[self.file['action/left_hand/x'][i], self.file['action/left_hand/y'][i], self.file['action/left_hand/z'][i], self.file['action/left_hand/roll'][i], self.file['action/left_hand/pitch'][i], self.file['action/left_hand/yaw'][i], self.file['action/left_hand/grasp'][i]] for i in range(0, idx+1)])
     
            else:
                observation =  torch.stack([self.transform(self.file['observation/image_head'][i]) for i in range(idx-self.window_size+1, idx+1)]).permute(0, 2, 3, 1)
                action = torch.tensor([[self.file['action/left_hand/x'][i], self.file['action/left_hand/y'][i], self.file['action/left_hand/z'][i], self.file['action/left_hand/roll'][i], self.file['action/left_hand/pitch'][i], self.file['action/left_hand/yaw'][i], self.file['action/left_hand/grasp'][i]] for i in range(idx-self.window_size+1, idx+1)])
                # image_wrists = torch.stack([self.transform(self.file['observation/image_wrist_left'][i]) for i in range(idx-self.window_size+1, idx+1)]).permute(0, 2, 3, 1)

            yield {'observation': {"image_primary": observation},
                                #    "image_wrist": sample['observation']['image_wrist']},
                'action': action,
                'language_instruction': torch.tensor(self.text_encoder([self.file['observation/language_instruction'][idx][0].decode()]).numpy())} 



# Define a PyTorch Lightning data module
class OXEDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32, data_dir: str = "/Users/sebastianperalta/simply/dev/simply-mono/robo_transformers/episodes/episode0.hdf5", shuffle: bool = False, seed: int = 42):
        super().__init__()
        self.batch_size = FLAGS.batch_size
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.seed = seed

    def setup(self, stage: str = None):
        # Load the dataset
        self.dataset = HD5Dataset(self.data_dir, shuffle=self.shuffle, seed=self.seed)

        self.train_ds = TorchRLDSDataset(*get_oxe_dataset(
            FLAGS.dataset_name,
            train=True,
            data_augmentation=FLAGS.data_augmentation,
            shuffle_buffer_size=FLAGS.shuffle_buffer_size))
        # self.dataset.prepare()

    def train_dataloader(self):
        # Create a PyTorch DataLoader
        return [DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=0
        ),
        DataLoader(
            self.train_ds,
            batch_size=FLAGS.batch_size,
            num_workers=
            0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
            pin_memory=True,
            # sampler= DistributedSampler(dataset=train_ds, shuffle=True) if torch.cuda.device_count() > 1 else None
        )]



class OXETrainingModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        
        action_space = spaces.Dict(
            OrderedDict(
                [
                    ("terminate_episode", spaces.Discrete(4)),
                    (
                        "world_vector",
                        spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                    ),
                    (
                        "rotation_delta",
                        spaces.Box(
                            low=-np.pi , high=np.pi, shape=(3,), dtype=np.float32
                        ),
                    ),
                    (
                        "gripper_closedness_action",
                        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                    ),
                ]
            )
        )

        args = {
            "mode": "train",
            "device": self.device,
            "cam_view": ["front"],
            "log_dir": "logs",
            "time_sequence_length": 6,
            "lr": 0.0001,
            "batch_size": FLAGS.batch_size,
            "epochs": 50,
            "resume": False,
            "resume_from_checkpoint": "",
            "predicting_next_ts": True,
            "world_size": 4,
            "val_interval": 1,
            "num_val_threads": 25,
            "num_train_episode": 200,
            "num_val_episode": 10,
            "using_proprioception": False,
            "network_configs": {
                "vocab_size": 256,
                "token_embedding_size_per_image": 512,
                "language_embedding_size": 512,
                "num_layers": 8,
                "layer_size": 128,
                "num_heads": 8,
                "feed_forward_size": 512,
                "dropout_rate": 0.1,
                "crop_size": 236,
                "use_token_learner": True,
            },
            "scheduler_configs": {"T_0": 10, "T_mult": 2, "eta_min": 1e-6, "verbose": True},
        }
        network_configs = args["network_configs"]
        # Modify network configuration based on specific settings
        network_configs["time_sequence_length"] = args["time_sequence_length"]
        network_configs["num_encoders"] = len(args["cam_view"])
        network_configs["token_embedding_size"] = network_configs[
            "token_embedding_size_per_image"
        ] * len(args["cam_view"])
        del network_configs["token_embedding_size_per_image"]
        network_configs["using_proprioception"] = args["using_proprioception"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = action_space
        self.model = TransformerNetwork(**network_configs)
        if FLAGS.weights_path:
            self.model.load_state_dict(torch.load(FLAGS.weights_path))
        self.action_tokenizer = ActionTokenizer()
        self.num_warmup_steps = FLAGS.num_warmup_steps
        self.num_decay_periods = 4
        self.total_steps = FLAGS.num_epochs * 1000
        self.max_lr = FLAGS.lr
        self.checkpoint_frequency = 100
        self.dirname = 'checkpoints/rtx_custom'
        self.accuracy = Accuracy(task='multiclass', num_classes=256).to(self.device)
        self.save_hyperparameters()
    


    def training_step(self, batch, batch_idx):
        self.model = self.model.to(self.device)
        self.accuracy = self.accuracy.to(self.device)
        # print('\n\n\nbatch action shape', batch['action'].shape)
        # training_step defines the train loop.
        video = rearrange(batch['observation']['image_primary'] * 1.0, 'b f h w c -> b f c h w').to(self.device)
        instructions = batch['language_instruction'].to(self.device)
        ground_truth = self.action_tokenizer.tokenize_xyzrpyg(
            batch['action'])[:,-1,:].to(self.device)
        
        obs = {'image': video, 'natural_language_embedding': repeat(instructions, 'b f n -> b (repeat f) n', repeat=video.shape[1])}

        # print('gt', ground_truth)
        # print('\n\nvideo og shaope:', video.shape)
        # print('nle', obs['natural_language_embedding'].shape)
        # exit()

        self.model.set_actions(dict_to_device({
            'terminate_episode': repeat(torch.ones((video.shape[0]), dtype=torch.long
                                                       ).to(self.device), 'b  -> b f', f=video.shape[1]),
            'world_vector':     batch['action'][:,:,0:3],
            'rotation_delta':   batch['action'][:,:,3:6],
            'gripper_closedness_action': batch['action'][:,:,6:]
        }, device=self.device))
        # model.set_actions(dict_to_device({
        #     'terminate_episode': torch.ones((video.shape[0], video.shape[1]), dtype=torch.long),
        #     'world_vector':     batch['action'][:,:,:3],
        #     'rotation_delta':   batch['action'][:,:,3:6],
        #     'gripper_closedness_action': batch['action'][:,:,6:]
        # }, device))
        network_state = np_to_tensor(
            batched_space_sampler(
                self.model._state_space,
                batch_size=video.shape[0],
            ),
            device=self.device
        )
        output_actions, network_state = self.model(
            dict_to_device(obs, self.device), 
              dict_to_device(network_state, self.device),
        )


        loss = self.model.get_actor_loss().mean()

        # if ground_truth[0,3] == 255 or ground_truth[0,3] == 0:
        #     loss = loss * 10.0

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # with torch.cuda.amp.autocast():

        # outs = reduce(model.train_step(video, instructions), 'b f a bins -> b a bins', 'mean')
        # out_preds = torch.max(outs, -1)[1]

        # loss = criterion(rearrange(outs, 'b a bins -> (b a) bins'),
        #                  rearrange(ground_truth, 'b a -> (b a)'))
        
        with torch.no_grad():
            out_preds = torch.cat([
                output_actions['world_vector'], 
                output_actions['rotation_delta'], 
                output_actions['gripper_closedness_action']], 1).to(self.device)
            out_preds = self.action_tokenizer.tokenize_xyzrpyg(out_preds.unsqueeze(1)).squeeze(1).to(self.device)
            # acc = (out_preds == ground_truth).float().mean()
            acc = self.accuracy(out_preds, ground_truth)
            # if self.global_step % 200 == 0:
                # print('acc', acc)
            self.log_dict(
                {
        
                    'x_pred_train': out_preds[0, 8],
                    'x_gt_train': ground_truth[0, 8],
                    'y_pred_train': out_preds[0, 9],
                    'y_gt_train': ground_truth[0, 9],
                    'z_pred_train': out_preds[0, 10],
                    'z_gt_train': ground_truth[0, 10],
                    'roll_pred_train': out_preds[0, 4],
                    'roll_gt_train': ground_truth[0, 4],
                    'pitch_pred_train': out_preds[0, 5],
                    'pitch_gt_train': ground_truth[0, 5],
                    'yaw_pred_train': out_preds[0, 6],
                    'yaw_gt_train': ground_truth[0, 6],
                    'grasp_pred_train': out_preds[0, 3],
                    'grasp_gt_train': ground_truth[0, 3],
                    # 'instruction_train': instructions[0],
                    'batch_idx': batch_idx,
                    'train_step': self.global_step,
                    'epoch': self.current_epoch,
                    'loss':  loss.item(),
                    'acc': acc,
                    'lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                #      'image_frames':  wandb.Video(
                #     np.array(255 * video[0, :, :, :, :].detach().to(
                #         'cpu')).astype(np.uint8),
                #     caption=
                #     f" {str(instructions[0])}, gt: {str(ground_truth[0,:])}, pred: {str(out_preds[0,:])}"
                # ),
                },
                on_step=True,
                prog_bar=False)
            self.log_dict(  {         
                    'loss_': loss.item(),
                    'acc_': acc,
                    'lr_': self.trainer.optimizers[0].param_groups[0]['lr'],
                    'train_step_': self.global_step,
                },on_step=True, prog_bar=True)
    # video = rearrange(batch['observation']['image_primary'],
    #                     'b f h w c -> b f c h w') * 1.0
    # instructions = batch['language_instruction']
    # ground_truth = self.action_tokenizer.tokenize_xyzrpyg(
    #     batch['action'])[:,-1,:]
        if self.global_step % 100 == 0:
                for i in range(video.shape[1]):
                    vide0 = video[0, i, :, :, :].detach().to('cpu')
                    video0 = rearrange(vide0, 'c h w -> h w c')
                    print('\n\n\n vide0: ', vide0.shape)
                    from PIL import Image
                    img = Image.fromarray((255 * video0.numpy()).astype(np.uint8))
                    img.save(f'./image{i}.png')


    # outs = self.model.train_step(video, instructions)[:,-1,:,:] # only take last frame
    # out_preds = torch.argmax(outs, -1)

    # loss = self.criterion(rearrange(outs, 'b a bins -> (b a) bins'),
    #                     rearrange(ground_truth, 'b a -> (b a)'))
        return loss



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.max_lr)
        # warmup_opt = optim.lr_scheduler.LinearLR(optimizer, total_iters=self.num_warmup_steps)

        # schedulers = []
        # for i in range(self.num_decay_periods):
        #     decaying_cos_opt = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer, T_0=1000, T_mult=2)
        #     schedulers.append(decaying_cos_opt)
        # decaying_cos_opt = optim.lr_schedulerCosineAnnealingWarmRestarts(
        #     optimizer, T_0=1000, T_mult=2)
        # scheduler = optim.lr_scheduler.SequentialLR(
        #     optimizer, [warmup_opt, decaying_cos_opt], [self.num_warmup_steps]
        # )
        lr_config = {
            'scheduler': optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=int(self.total_steps)),
            'interval': 'step',
            'frequency': 1
        }
        # return [optimizer], [scheduler]
        return [optimizer], [lr_config]


def main(_):
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="rtdiffusion",
    #     config=dict(num_epochs=FLAGS.num_epochs,
    #                 batch_size=FLAGS.batch_size,
    #                 num_warmup_steps=FLAGS.num_warmup_steps,
    #                 shuffle_buffer_size=FLAGS.shuffle_buffer_size,
    #                 eval_batch_size=FLAGS.eval_batch_size,
    #                 lr=FLAGS.lr,
    #                 min_lr=FLAGS.min_lr,
    #                 weight_decay=FLAGS.weight_decay,
    #                 dataset_name=FLAGS.dataset_name,
    #                 checkpoint_dir=FLAGS.checkpoint_dir,
    #                 baselines=FLAGS.baselines,
    #                 data_augmentation=FLAGS.data_augmentation,
    #                 conditioning_scale=FLAGS.conditioning_scale,
    #                 label_smoothing=FLAGS.label_smoothing,
    #                 loss=FLAGS.loss,
    #                 freeze_vit=FLAGS.freeze_vit))

    # Initialize the Lightning Module
# model = LitModel((3, 32, 32), dm.num_classes)

# # Initialize wandb logger
# wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

# # Initialize Callbacks
# early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
# checkpoint_callback = pl.callbacks.ModelCheckpoint()

# # Initialize a trainer
# trainer = pl.Trainer(max_epochs=2,
#                      logger=wandb_logger,
#                      callbacks=[early_stop_callback,
#                                 ImagePredictionLogger(val_samples),
#                                 checkpoint_callback],
#                      )

# # Train the model ⚡🚅⚡
# trainer.fit(model, dm)

# # Evaluate the model on the held-out test set ⚡⚡
# trainer.test(dataloaders=dm.test_dataloader())

# # Close wandb run
# wandb.finish()    
    wandb.login()
    wandb_logger = WandbLogger(project='rt1diffusion', job_type='train')

    module = OXETrainingModule()

    dataset_path = files('oxe_torch').joinpath('episodes/episode0.hdf5')
    data_module = OXEDataModule(data_dir=dataset_path)
    # Initialize the Lightning Trainer
    if torch.cuda.is_available():
        accelerator = 'auto'
    else:
        accelerator = 'cpu'
    trainer = L.Trainer(max_steps=1e5, logger=wandb_logger, callbacks=[CHECKPOINT_CALLBACK], accelerator=accelerator)
    # trainer = L.Trainer(fast_dev_run=True, accelerator=accelerator)
    # Start the training loop
    trainer.fit(module, datamodule=data_module)
# class OXETrainingModule(L.LightningModule):
#     def training_step(self, batch, batch_idx):
#         # training_step defines the train loop.
        
#         # Log weights and biases
#         self.logger.log_metrics({"weight": self.encoder.weight.item(), "bias": self.encoder.bias.item()}, step=self.global_step)
        
#         # Rest of your training code
        
#         return loss

if __name__ == "__main__":
    app.run(main)

