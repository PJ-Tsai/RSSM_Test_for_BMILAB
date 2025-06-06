# RSSM-based Delta Pose Predictor for Ultrasound Images
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import csv
import cv2
import torch.optim as optim
import torch.nn.functional as F

# Set Data Root
data_root = '.'  # /path/to/data

# Set train/test folder names
train_dirs = {
    "frames_0513_06", "frames_0513_07", "frames_0513_08", "frames_0513_09",
    "frames_0513_11", "frames_0513_12", "frames_0513_13", "frames_0513_14",
    "frames_0513_16", "frames_0513_17", "frames_0513_18", "frames_0513_19", "frames_0513_20",
    "frames_0513_21", "frames_0513_22", "frames_0513_23", "frames_0513_24", "frames_0513_25", "frames_0513_26"
}

test_dirs = {
    "frames_0513_01", "frames_0513_02", "frames_0513_03", "frames_0513_04", "frames_0513_05"
}

# ========= Combine CSVs ============
def combine_pose_csvs_with_foldername(root_folder, output_csv="poses_combined.csv"):
    all_data = []

    for file in sorted(os.listdir(root_folder)):
        if not file.endswith("_final_data.csv"):
            continue

        csv_path = os.path.join(root_folder, file)
        df = pd.read_csv(csv_path)

        # ex: 0513_01_final_data.csv → frames_0513_01
        folder_name = "frames_" + file.replace("_final_data.csv", "")

        # Update Filename Column: → frames_0513_01/frame_0000.png
        df["Filename"] = df["Filename"].apply(lambda x: f"{folder_name}/{x}")
        all_data.append(df)

    if not all_data:
        print("⚠️ No Valid File Found")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"✅ Saved Combined CSV to：{output_csv}")
    
combine_pose_csvs_with_foldername(data_root, "poses_combined.csv") 

# === Inference Ultrasound Dataset No Goal ===
class InferenceUltrasoundDatasetNoGoal(Dataset):
    def __init__(self, csv_path, root_dirs, init_len=5, inf_len=2, image_size=(256, 256)):
        self.samples = []
        self.inf_len = inf_len
        df = pd.read_csv(csv_path)
        df['folder'] = df['Filename'].apply(lambda x: x.split('/')[0])
        df = df.rename(columns={
            'Filename': 'img_path',
            'X (mm)': 'tx', 'Y (mm)': 'ty', 'Z (mm)': 'tz',
            'Roll (deg)': 'rx', 'Pitch (deg)': 'ry', 'Yaw (deg)': 'rz'
        })

        for dir_ in root_dirs:
            group = df[df['folder'] == dir_].sort_values('img_path').reset_index(drop=True)
            frames = group.to_dict('records')
            num_frames = len(frames)
            if num_frames <= init_len + inf_len:
                continue

            # generate (init_seq, inf_seq) pair
            for start_idx in range(0, num_frames - init_len - inf_len + 1):
                init_seq = frames[start_idx : start_idx + init_len]
                inf_seq = frames[start_idx + init_len : start_idx + init_len + inf_len]
                sample = {
                    'init_sequence': init_seq,
                    'inference_sequence': inf_seq,
                    'sequence_name': dir_
                }
                self.samples.append(sample)

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def _load_image(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        return self.transform(img)

    def __getitem__(self, idx):
        item = self.samples[idx]

        init_imgs = [self._load_image(x['img_path']) for x in item['init_sequence']]
        inf_imgs = [self._load_image(x['img_path']) for x in item['inference_sequence']]

        inf_poses = [
            np.array([
                x['tx'], x['ty'], x['tz'],
                x['rx'], x['ry'], x['rz']
            ], dtype=np.float32)
            for x in item['inference_sequence']
        ]

        delta_poses = np.diff(np.stack(inf_poses, axis=0), axis=0)  # [T-1, 6]

        return {
            'sequence_name': item['sequence_name'],
            'init_images': torch.stack(init_imgs),                   # [init_len, 1, H, W]
            'inference_images': torch.stack(inf_imgs),               # [T, 1, H, W]
            'ground_truth_delta_poses': torch.tensor(delta_poses, dtype=torch.float32),  # [T-1, 6]
        }

# === Encoder ===
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.InstanceNorm2d(32), nn.ReLU(),  # 256→128
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU(),  # 128→64
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(),  # 64→32
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU()  # 32→16
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = F.softplus(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

# === RSSM Core ===
class RSSMCore(nn.Module):
    def __init__(self, action_dim, z_dim, h_dim, embed_dim):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.project_action_z = nn.Linear(z_dim + action_dim, h_dim)
        self.gru = nn.GRUCell(h_dim, h_dim)

        self.project_hidden_action = nn.Linear(h_dim + action_dim, h_dim)
        self.prior = nn.Linear(h_dim, z_dim * 2)

        self.project_hidden_obs = nn.Linear(h_dim + embed_dim, h_dim)
        self.posterior = nn.Linear(h_dim, z_dim * 2)

        self.activation = nn.ReLU()

    def forward(self, prev_z, prev_h, actions, embeddings=None, dones=None):
        B, T, _ = actions.size()
        h, z = prev_h, prev_z

        h_seq, z_seq, prior_mean_seq, prior_std_seq = [], [], [], []
        post_mean_seq, post_std_seq = [], []
        # min_std = 1e-3  # can be adjusted

        for t in range(T):
            a = actions[:, t]
            e = embeddings[:, t] if embeddings is not None else None

            # Reset z if done
            if dones is not None:
                z = z * (1.0 - dones[:, t])

            x = torch.cat([z, a], dim=-1)
            x = self.activation(self.project_action_z(x))
            h = self.gru(x, h)

            # Prior
            ha = torch.cat([h, a], dim=-1)
            ha = self.activation(self.project_hidden_action(ha))
            prior_params = self.prior(ha)
            prior_mean, prior_logstd = torch.chunk(prior_params, 2, dim=-1)
            prior_std = F.softplus(prior_logstd) #+ min_std
            prior_dist = torch.distributions.Normal(prior_mean, prior_std)
            prior_z = prior_dist.rsample()

            # Posterior
            if embeddings is not None:
                he = torch.cat([h, e], dim=-1)
                he = self.activation(self.project_hidden_obs(he))
                post_params = self.posterior(he)
                post_mean, post_logstd = torch.chunk(post_params, 2, dim=-1)
                post_std = F.softplus(post_logstd) #+ min_std
                post_dist = torch.distributions.Normal(post_mean, post_std)
                post_z = post_dist.rsample()
            else:
                post_z = prior_z
                post_mean, post_std = prior_mean, prior_std

            z = post_z

            # Collect for each timestep
            h_seq.append(h.unsqueeze(1))
            z_seq.append(z.unsqueeze(1))
            prior_mean_seq.append(prior_mean.unsqueeze(1))
            prior_std_seq.append(prior_std.unsqueeze(1))
            post_mean_seq.append(post_mean.unsqueeze(1))
            post_std_seq.append(post_std.unsqueeze(1))

        return {
            'h': torch.cat(h_seq, dim=1),
            'z': torch.cat(z_seq, dim=1),
            'prior_mean': torch.cat(prior_mean_seq, dim=1),
            'prior_std': torch.cat(prior_std_seq, dim=1),
            'post_mean': torch.cat(post_mean_seq, dim=1),
            'post_std': torch.cat(post_std_seq, dim=1),
        }
        
    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.z_dim, device=device),
            torch.zeros(batch_size, self.h_dim, device=device)
        )

    def step(self, prev_z, prev_h, action, embedding=None, done=None):
        x = torch.cat([prev_z, action], dim=-1)
        x = self.activation(self.project_action_z(x))
        h = self.gru(x, prev_h)

        ha = torch.cat([h, action], dim=-1)
        ha = self.activation(self.project_hidden_action(ha))
        prior_params = self.prior(ha)
        prior_mean, prior_logstd = torch.chunk(prior_params, 2, dim=-1)
        prior_std = F.softplus(prior_logstd)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_z = prior_dist.rsample()

        if embedding is not None:
            he = torch.cat([h, embedding], dim=-1)
            he = self.activation(self.project_hidden_obs(he))
            post_params = self.posterior(he)
            post_mean, post_logstd = torch.chunk(post_params, 2, dim=-1)
            post_std = F.softplus(post_logstd)
            post_dist = torch.distributions.Normal(post_mean, post_std)
            post_z = post_dist.rsample()
        else:
            post_z = prior_z
            post_mean, post_std = prior_mean, prior_std

        if done is not None:
            post_z = post_z * (1.0 - done)

        return h, post_z, post_mean, post_std


# === Pose Decoder ===
class PoseDecoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, 128), nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, h):
        return self.fc(h)

# === Frame Decoder ===
class FrameDecoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.fc = nn.Linear(h_dim, 128 * 16 * 16)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1)
        )

    def forward(self, h):
        x = self.fc(h).view(-1, 128, 16, 16)
        x = self.deconv(x)
        return x

class RSSMGoalDeltaPoseModel(nn.Module):
    def __init__(self, z_dim=64, h_dim=256, action_dim=6, embed_dim=64):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.rssm = RSSMCore(action_dim, z_dim, h_dim, embed_dim)
        self.pose_decoder = PoseDecoder(h_dim)
        self.frame_decoder = FrameDecoder(h_dim)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.action_dim = action_dim

    def forward(self, init_imgs, inf_imgs):  
        """
        init_imgs: [B, init_len, 1, H, W]
        inf_imgs:  [B, T, 1, H, W]
        """
        B, init_len, _, H, W = init_imgs.shape
        T = inf_imgs.shape[1]
        device = init_imgs.device

        z, h = self.rssm.init_hidden(B, device)

        # --- Encode and initialize with init sequence ---
        init_embeds, _, _ = self.encoder(init_imgs.view(B * init_len, 1, H, W))
        init_embeds = init_embeds.view(B, init_len, -1)

        zero_action = torch.zeros(B, self.action_dim, device=device)
        for t in range(init_len):
            h, z, _, _ = self.rssm.step(z, h, zero_action, embedding=init_embeds[:, t])

        # --- Encode inference images ---
        inf_embeds, mus, logvars = [], [], []
        for t in range(T):
            embed, mu, logvar = self.encoder(inf_imgs[:, t])
            inf_embeds.append(embed.unsqueeze(1))
            mus.append(mu.unsqueeze(1))
            logvars.append(logvar.unsqueeze(1))

        inf_embeds = torch.cat(inf_embeds, dim=1)  # [B, T, z_dim]
        mus = torch.cat(mus, dim=1)
        logvars = torch.cat(logvars, dim=1)

        # --- Rollout using inference embeddings ---
        pred_delta_poses = []
        recon_imgs = []
        prev_action = torch.zeros(B, self.action_dim, device=device)

        for t in range(T):
            h, z, _, _ = self.rssm.step(z, h, prev_action, embedding=inf_embeds[:, t])
            delta_pose = self.pose_decoder(h)       # [B, 6]
            recon_img = self.frame_decoder(h)       # [B, 1, H, W]
            pred_delta_poses.append(delta_pose)
            recon_imgs.append(recon_img)
            prev_action = delta_pose.detach()

        pred_delta_poses = torch.stack(pred_delta_poses, dim=1)  # [B, T, 6]
        recon_imgs = torch.stack(recon_imgs, dim=1)              # [B, T, 1, H, W]
        kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=-1).mean()

        return pred_delta_poses, recon_imgs, kl_loss

@torch.no_grad()
def batched_predict_sequence(model, init_batch, inf_batch, device):
    """
    Batched inference over multiple sequences (batch_size > 1)
    - init_batch: [B, init_len, 1, H, W]
    - inf_batch:  [B, T, 1, H, W]

    Returns:
        predicted_delta_poses: [B, T-1, 6]
    """
    model.eval()
    B, init_len, _, H, W = init_batch.shape
    T = inf_batch.shape[1]  # T == inference_len

    encoder = model.encoder
    rssm = model.rssm
    pose_decoder = model.pose_decoder

    init_batch = init_batch.to(device)
    inf_batch = inf_batch.to(device)

    # --- Initialize latent state ---
    z, h = rssm.init_hidden(B, device)

    # --- Feed init images to accumulate state ---
    init_images = init_batch.view(B * init_len, 1, H, W)
    init_embeds, _, _ = encoder(init_images)
    init_embeds = init_embeds.view(B, init_len, -1)

    # Use zero action only during init
    zero_action = torch.zeros(B, model.action_dim, device=device)
    for t in range(init_len):
        h, z, _, _ = rssm.step(z, h, zero_action, embedding=init_embeds[:, t])

    # --- Predict delta poses using inference images ---
    inf_images = inf_batch[:, :-1].reshape(B * (T - 1), 1, H, W)
    inf_embeds, _, _ = encoder(inf_images)
    inf_embeds = inf_embeds.view(B, T - 1, -1)

    pred_delta_poses = []
    prev_action = torch.zeros(B, model.action_dim, device=device)  # or learned init

    for t in range(T - 1):
        h, z, _, _ = rssm.step(z, h, prev_action, embedding=inf_embeds[:, t])
        delta_pose = pose_decoder(h)  # [B, 6]
        pred_delta_poses.append(delta_pose)
        prev_action = delta_pose.detach()  # Use prediction as next action

    pred_delta_poses = torch.stack(pred_delta_poses, dim=1)  # [B, T-1, 6]
    return pred_delta_poses

# === Weighted_pose_loss ===
def weighted_pose_loss(pred, target, lambda_sign=1000.0, return_components=False):
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=pred.device)

    scale_factor=1000.0
    weighted_pred = pred * weights * scale_factor
    weighted_target = target * weights * scale_factor
    base_loss = F.smooth_l1_loss(weighted_pred, weighted_target)

    # sign consistency loss
    sign_penalty = torch.relu(-pred * target)  # penalize different signs
    sign_loss = (sign_penalty * weights).mean()
    sign_loss *= lambda_sign

    total_loss = base_loss + sign_loss
    
    if return_components:
        return total_loss, base_loss, sign_loss
    else:
        return total_loss

# === Signed Sqrt and Square ===
def signed_sqrt(x, eps=1e-8):
    return torch.sign(x) * torch.sqrt(torch.abs(x) + eps)

def signed_square(x):
    return torch.sign(x) * (x ** 2)

# === Train Model ===
def train_model(csv_path, train_dirs, test_dirs, image_size=(256, 256), batch_size=8, epochs=20, lr=1e-4):
    loss_log_path = 'RSSM_7_8_losses.csv'
    with open(loss_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss', 'train_base_loss', 'train_sign_loss', 'train_recon_loss', 'train_kl_loss',
            'test_loss',
            'acc_tx', 'acc_ty', 'acc_tz', 'acc_rx', 'acc_ry', 'acc_rz', 'overall_acc'
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = InferenceUltrasoundDatasetNoGoal(csv_path, train_dirs, init_len=5, inf_len=2, image_size=image_size)
    test_dataset = InferenceUltrasoundDatasetNoGoal(csv_path, test_dirs, init_len=5, inf_len=2, image_size=image_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = RSSMGoalDeltaPoseModel(h_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_train_loss = float('inf')

    for epoch in range(epochs):
        # ==== Training ====
        model.train()
        total_train_loss = total_train_base_loss = total_train_sign_loss = 0.0
        total_train_recon_loss = total_train_kl_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            init_imgs = batch['init_images'].to(device)                    # [B, init_len, 1, H, W]
            inf_imgs = batch['inference_images'].to(device)                # [B, inf_len, 1, H, W]
            delta_poses = batch['ground_truth_delta_poses'].to(device)     # [B, inf_len-1, 6]
    
            optimizer.zero_grad()
    
            pose_preds, img_preds, kl_loss = model(init_imgs, inf_imgs[:, :-1])

            sqrt_gt = signed_sqrt(delta_poses)    
            pose_loss, base_loss, sign_loss = weighted_pose_loss(pose_preds, sqrt_gt, lambda_sign=1000.0, return_components=True)
            recon_loss = F.mse_loss(img_preds, inf_imgs[:, 1:])
    
            loss = 5.0 * pose_loss + 1.0 * recon_loss + min(0.5, epoch/20) * kl_loss # warmup epoch = 10
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_base_loss += base_loss.item()
            total_train_sign_loss += sign_loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_kl_loss += kl_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_base_loss = total_train_base_loss / len(train_loader)
        avg_train_sign_loss = total_train_sign_loss / len(train_loader)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_kl_loss = total_train_kl_loss / len(train_loader)

        # ==== Testing ====
        model.eval()
        total_test_loss = 0.0
        all_test_pred = []
        all_test_true = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1} Eval"):
                # batch keys: 'sequence_name', 'image_sequence', 'inference_images', 'ground_truth_delta_poses'
                init_images = batch['init_images'].to(device)           # [B, init_len, 1, H, W]
                inf_images = batch['inference_images'].to(device)       # [B, T, 1, H, W]
                target_delta_pose = batch['ground_truth_delta_poses'].to(device)  # [B, T-1, 6]

                pred_delta_pose = batched_predict_sequence(model, init_images, inf_images, device)

                sqrt_target_delta_pose = signed_sqrt(target_delta_pose)
                loss = weighted_pose_loss(pred_delta_pose, sqrt_target_delta_pose)
                pred_delta_pose = signed_square(pred_delta_pose)

                total_test_loss += loss.item()  # batch=1
                
                all_test_pred.append(pred_delta_pose.cpu().numpy())
                all_test_true.append(target_delta_pose.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_loader)

        # ==== Accuracy Evaluation ====
        all_test_pred_arr = np.concatenate(all_test_pred, axis=0).astype(np.float32).reshape(-1, 6)
        all_test_true_arr = np.concatenate(all_test_true, axis=0).astype(np.float32).reshape(-1, 6)

        pred = all_test_pred_arr  # shape: [N, 6]
        true = all_test_true_arr
        epsilon = 1e-6
        within_bounds = (pred >= 0.5 * (true + epsilon)) & (pred <= 1.5 * (true + epsilon))
        true_zero = np.abs(true) < epsilon
        pred_zero = np.abs(pred) < epsilon
        zero_match = true_zero & pred_zero
        correct_mask = within_bounds | zero_match

        component_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        component_accuracies = {
            name: correct_mask[:, i].mean()
            for i, name in enumerate(component_names)
        }
        overall_accuracy = np.all(correct_mask, axis=1).mean()
        
        # Save best model
        """
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), 'RSSM_7_8_best_model.pth')
            print(f"✅ Best model saved at epoch {epoch+1} with test_loss: {avg_train_loss:.6f}")
        """
        # Save test results CSV
        ratio = (pred + epsilon) / (true + epsilon)
        df_test_results = pd.DataFrame({
            **{f'pred_{name}': pred[:, i].flatten() for i, name in enumerate(component_names)},
            **{f'true_{name}': true[:, i].flatten() for i, name in enumerate(component_names)},
            **{f'ratio_{name}': ratio[:, i].flatten() for i, name in enumerate(component_names)},
        })
        df_test_results.to_csv(f'RSSM_7_8_test_pred_true_epoch_{epoch+1}.csv', index=False)
        
        # Print metrics
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        for name in component_names:
            print(f"  {name} acc: {component_accuracies[name]:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        
        # Save model and log
        torch.save(model.state_dict(), f'RSSM_7_8_model_epoch_{epoch+1}.pth')
        # Log losses
        with open(loss_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss, avg_train_base_loss, avg_train_sign_loss, avg_train_recon_loss, avg_train_kl_loss,
                avg_test_loss,
                component_accuracies['tx'], component_accuracies['ty'], component_accuracies['tz'],
                component_accuracies['rx'], component_accuracies['ry'], component_accuracies['rz'],
                overall_accuracy
            ])
            
# === Run ===
if __name__ == "__main__":
    train_model(
        csv_path="poses_combined.csv",
        train_dirs=train_dirs,
        test_dirs=test_dirs,
        epochs=60,
        batch_size=16,
        lr=1e-4,
        image_size=(256, 256)
    )