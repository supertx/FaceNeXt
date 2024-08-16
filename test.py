from models import FaceMAE
from criterion import PretrainLoss, DiscriminatorLoss, Discriminator
from utils import get_config
from dataset import getDataloader

cfg = get_config("pretrain_FaceNeXt_tiny.yml", is_pretrain=True)

dataloader = getDataloader(cfg.dataset.data_dir, cfg.dataset.batch_size,
                           num_workers=cfg.dataset.num_workers, is_train=True)

model = FaceMAE(depth=cfg.model.depth,
                dims=cfg.model.dims,
                decoder_depth=cfg.model.decoder_depth,
                patch_size=cfg.dataset.patch_size,
                mask_ratio=cfg.model.mask_ratio,
                inner_scale=cfg.model.inner_scale)
discriminator = Discriminator()
pretrain_loss = PretrainLoss(cfg, discriminator)
discriminator_loss = DiscriminatorLoss(discriminator)
model.cuda()
discriminator.cuda()


for img, anno, _ in dataloader:
    img = img.cuda()
    anno = anno.cuda()
    pred, mask = model(img, anno)
    loss = pretrain_loss(img, pred, mask)
    loss.backward()
    loss = discriminator_loss(pred)
    loss.backward()
    break