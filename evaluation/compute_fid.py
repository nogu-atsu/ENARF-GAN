import argparse
import warnings

from cleanfid.fid import *
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
from dependencies.config import yaml_config
from dataset import THUmanDataset, THUmanPoseDataset, HumanDataset, HumanPoseDataset
from models.generator import NARFNRGenerator, TriNARFGenerator

warnings.filterwarnings('ignore')


class GenIterator:
    def __init__(self, gen, dataloader, num_sample, black_bg_if_possible=False):
        self.gen = gen
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.num_sample = num_sample
        self.i = 0
        self.black_bg_if_possible = black_bg_if_possible
        assert len(dataloader.dataset) > 0
        assert len(dataloader.dataset) >= num_sample

        self.data = self.generator()

        self.gen.train()  # avoid fixed cropping of background

    def __iter__(self):
        return self

    def __len__(self):
        return (self.num_sample - 1) // self.batch_size + 1

    def generator(self):
        for minibatch in self.dataloader:
            yield minibatch

    def __next__(self):
        if self.i == len(self):
            raise StopIteration()
        minibatch = self.data.__next__()  # randomly sample latent

        batchsize = len(minibatch["pose_to_camera"])

        z_dim = self.gen.config.z_dim * 3 if is_VAE else self.gen.config.z_dim * 4
        z = torch.cuda.FloatTensor(batchsize, z_dim).normal_()

        pose_to_camera = minibatch["pose_to_camera"].cuda(non_blocking=True)
        bone_length = minibatch["bone_length"].cuda(non_blocking=True)
        pose_to_world = minibatch["pose_to_world"].cuda(non_blocking=True)
        intrinsic = minibatch["intrinsics"].cuda(non_blocking=True)
        inv_intrinsic = torch.inverse(intrinsic)
        with torch.no_grad():
            fake_img, _, _, _ = self.gen(pose_to_camera, pose_to_world, bone_length, z, inv_intrinsic,
                                         black_bg_if_possible=self.black_bg_if_possible,
                                         truncation_psi=args.truncation)
        self.i += 1
        return torch.clamp(fake_img, -1, 1)


def data_iterator(loader, n_batch):
    for i, data in enumerate(loader):
        yield data["img"].float()
        if i == (n_batch - 1):
            break


"""
Compute the FID stats from a gener
"""


def get_model_features_from_imgs(data_iterator, model, mode="clean", batch_size=128,
                                 device=torch.device("cuda"), desc="FID model: "):
    # compute features from
    fn_resize = build_resizer(mode)
    # Generate test features
    l_feats = []
    for img_batch in tqdm(data_iterator, desc=desc):
        with torch.no_grad():
            # color range is [0, 255]
            img_batch = img_batch * 127.5 + 127.5

            # split into individual batches for resizing if needed
            resized_batch = F.interpolate(img_batch, (299, 299), mode="bilinear")
            # if mode != "legacy_tensorflow":
            #     resized_batch = torch.zeros(batch_size, 3, 299, 299)
            #     for idx in range(len(img_batch)):
            #         curr_img = img_batch[idx]
            #         img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
            #         img_resize = fn_resize(img_np)
            #         resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)))
            # else:
            #     resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


def load_statistics(config, feat_model, mode, batch_size, device, desc, num_sample):
    data_root = config.train.data_root

    # black_bg is supported for surreal
    if args.black_bg:
        assert "SURREAL" in data_root
        data_root = "/data/unagi0/noguchi/dataset/SURREAL/SURREAL/data/cmu/NARF_GAN_segmented_cache"

    # statistics are already computed
    mu_path = f"{data_root}/fid_statistics/mu_{num_sample}.npy"
    sigma_path = f"{data_root}/fid_statistics/sigma_{num_sample}.npy"
    if os.path.exists(mu_path):
        mu = np.load(mu_path)
        sigma = np.load(sigma_path)
        return mu, sigma

    # return saved values
    dataset_name = config.name
    size = config.image_size
    just_cache = False

    train_dataset_config = config.train

    print("loading datasets")
    if dataset_name == "human":
        img_dataset = THUmanDataset(train_dataset_config, size=size, return_bone_params=False,
                                    just_cache=just_cache)

    elif dataset_name == "human_v2":
        # TODO mixed prior
        img_dataset = HumanDataset(train_dataset_config, size=size, return_bone_params=False,
                                   just_cache=just_cache)
        pose_prior_root = train_dataset_config.pose_prior_root or train_dataset_config.data_root
        print("pose prior:", pose_prior_root)
    else:
        raise ValueError()
    img_dataset.num_repeat_in_epoch = 1
    loader_img = DataLoader(img_dataset, batch_size=batch_size, num_workers=2, shuffle=True,
                            drop_last=False)
    loader_img = data_iterator(loader_img, num_sample // batch_size)
    mu, sigma = calc_statistics(loader_img, feat_model, mode, batch_size, device, desc)

    # save statistics
    os.makedirs(f"{config.train.data_root}/fid_statistics", exist_ok=True)
    np.save(mu_path, mu)
    np.save(sigma_path, sigma)
    return mu, sigma


def calc_statistics(iterator, feat_model, mode, batch_size, device, desc):
    np_feats = get_model_features_from_imgs(iterator, feat_model, mode, batch_size, device, desc)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    return mu, sigma


def my_fid_func(config, mode="legacy_pytorch", batch_size=4, num_sample=10_000,
                device=torch.device("cuda"), desc="FID model: "):
    size = config.dataset.image_size
    dataset_name = config.dataset.name
    train_dataset_config = config.dataset.train

    just_cache = False

    print("loading datasets")
    if dataset_name == "human":
        pose_prior_root = train_dataset_config.data_root

        print("pose prior:", pose_prior_root)
        pose_dataset = THUmanPoseDataset(size=size, data_root=pose_prior_root,
                                         just_cache=just_cache)
    elif dataset_name == "human_v2":
        pose_prior_root = train_dataset_config.data_root
        print("pose prior:", pose_prior_root)
        pose_dataset = HumanPoseDataset(size=size, data_root=pose_prior_root,
                                        just_cache=just_cache)


    else:
        assert False
    # pose_dataset.num_repeat_in_epoch = 1
    loader_pose = DataLoader(pose_dataset, batch_size=batch_size, num_workers=2, shuffle=True,
                             drop_last=True)

    gen_config = config.generator_params

    if gen_config.use_triplane:
        gen = TriNARFGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                               num_bone_param=pose_dataset.num_bone_param,
                               parent_id=pose_dataset.parents,
                               black_background=is_VAE)
        gen.register_canonical_pose(pose_dataset.canonical_pose)
        gen.to("cuda")
    else:
        gen = NARFNRGenerator(gen_config, size, num_bone=pose_dataset.num_bone,
                              num_bone_param=pose_dataset.num_bone_param, parent_id=pose_dataset.parents).to("cuda")

    out_dir = config.out_root
    out_name = config.out
    iteration = args.iteration if args.iteration > 0 else "latest"
    path = f"{out_dir}/result/{out_name}/snapshot_{iteration}.pth"
    if os.path.exists(path):
        snapshot = torch.load(path, map_location="cuda")
        for k in list(snapshot["gen"].keys()):
            if "activate.bias" in k:
                snapshot["gen"][k[:-13] + "bias"] = snapshot["gen"][k].reshape(1, -1, 1, 1)
                del snapshot["gen"][k]
        gen.load_state_dict(snapshot["gen"], strict=False)
    else:
        assert False, "pretrained model is not loading"

    gen = GenIterator(gen, loader_pose, num_sample=num_sample, black_bg_if_possible=args.black_bg)

    feat_model = build_feature_extractor(mode, device)

    ref_mu, ref_sigma = load_statistics(config.dataset, feat_model, mode, batch_size, device, desc, num_sample)

    mu, sigma = calc_statistics(gen, feat_model, mode, batch_size, device, desc)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)

    if args.truncation != 1:
        suffix = f"_trunc{args.truncation}"
    else:
        suffix = ""
    if args.black_bg:
        print(args.config, "black fid:", fid)
        with open(f"{out_dir}/result/{out_name}/black_fid{suffix}.txt", "w") as f:
            f.write(f"{fid}")
    else:
        print(args.config, "fid:", fid)
        with open(f"{out_dir}/result/{out_name}/fid{suffix}.txt", "w") as f:
            f.write(f"{fid}")

    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/NARF_GAN/THUman/20210903.yml")
    parser.add_argument('--default_config', type=str, default="configs/NARF_GAN/default.yml")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--black_bg', action="store_true")
    parser.add_argument('--truncation', type=float, default=1)

    args = parser.parse_args()

    config = yaml_config(args.config, args.default_config, num_workers=args.num_workers)
    is_VAE = "VAE" in args.config

    my_fid_func(config)
