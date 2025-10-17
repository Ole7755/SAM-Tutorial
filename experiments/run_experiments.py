import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd
import json
import os
from pathlib import Path


# ==================== 配置类 ====================
class AttackConfig:
    """实验配置"""

    def __init__(self):
        # 模型配置
        self.model_type = "vit_l"
        self.checkpoint = "sam_vit_l_0b3195.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 图像和提示
        self.image_path = self._get_project_relative_path("cat.png")
        self.point_coords = None  # 运行时根据图像大小设置
        self.point_label = 1  # 前景点

        # FGSM配置
        self.fgsm_epsilon = [4 / 255, 8 / 255, 16 / 255]

        # PGD配置
        self.pgd_epsilon = 8 / 255
        self.pgd_alpha = 2 / 255
        self.pgd_steps_list = [7, 10, 20]

        # 损失函数配置
        self.loss_types = ["minimize_area", "minimize_score", "bce_loss", "combined"]
        self.combined_alpha = 1.0
        self.combined_beta = 10.0

        # 输出配置
        self.results_dir = self._get_project_relative_path("results")
        self.figures_dir = self._get_project_relative_path("results", "figures")
        self.logs_dir = self._get_project_relative_path("results", "logs")
        self.setup_directories()

    def _get_project_relative_path(self, *path_parts):
        """获取相对于项目根目录的路径"""
        current_file_dir = Path(os.path.abspath(__file__)).parent
        return str(current_file_dir.parent.joinpath(*path_parts))

    def setup_directories(self):
        """确保所有输出目录都存在 - 自动创建缺失的目录"""
        directories = {
            "Results": self.results_dir,
            "Figures": self.figures_dir,
            "Logs": self.logs_dir,
        }

        for name, path in directories.items():
            try:
                path_obj = Path(path)

                # 先检查是否存在
                if path_obj.exists():
                    # 额外检查是否是目录
                    if not path_obj.is_dir():
                        raise RuntimeError(
                            f"{path} exists but is a file, not a directory"
                        )
                    status = "[EXISTS]"
                    action = "already exists"

                else:
                    # 如果文件夹不存在,则直接创建
                    path_obj.mkdir(parents=True, exist_ok=True)
                    status = "[CREATED]"
                    action = "created"

                print(f"{status} {name} directory {action}: {path}")

            except PermissionError as e:
                print(f"[ERROR] Permission denied for {name} directory '{path}': {e}")
                raise RuntimeError(f"No permission to create directory: {path}") from e
            except Exception as e:
                print(f"[ERROR] Failed to create {name} directory '{path}': {e}")
                raise RuntimeError(f"Directory creation failed for {path}") from e


# ==================== 损失函数 ====================
class LossFunction:
    """四种损失函数实现 - 针对 'Segment Nothing' 目标"""

    @staticmethod
    def minimize_area(logits, scores=None):
        """
        Loss 1: 最小化Mask面积
        直觉: 让mask中的激活像素尽可能少
        """
        mask_probs = torch.sigmoid(logits)
        loss = -mask_probs.sum()
        return loss

    @staticmethod
    def minimize_score(logits, scores):
        """
        Loss 2: 最小化SAM的质量分数
        直觉: 利用SAM自身的质量评估,让模型认为分割很差
        """
        if scores is None:
            return torch.tensor(0.0, device=logits.device)
        loss = -scores.mean()
        return loss

    @staticmethod
    def bce_loss(logits, scores=None):
        """
        Loss 3: 二元交叉熵 (目标为空mask)
        直觉: 显式地将目标设为全0 mask
        """
        target = torch.zeros_like(logits)
        loss = -F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        return loss

    @staticmethod
    def combined_loss(logits, scores, alpha=1.0, beta=10.0):
        """
        Loss 4: 组合损失
        直觉: 同时优化mask面积和质量分数
        """
        area_loss = torch.sigmoid(logits).sum()
        score_loss = scores.mean() if scores is not None else 0
        loss = -(alpha * area_loss + beta * score_loss)
        return loss


# ==================== SAM攻击器 ====================


class SAMAttacker:
    """SAM对抗攻击实现 - 支持FGSM和PGD"""

    def __init__(self, config: AttackConfig):
        self.config = config

        # 加载SAM模型
        print(f"\n{'='*70}")
        print(f"Loading SAM Model")
        print(f"{'='*70}")
        print(f"Model type: {config.model_type}")
        print(f"Checkpoint: {config.checkpoint}")

        sam = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
        sam.to(device=config.device)
        sam.eval()

        self.sam = sam
        self.predictor = SamPredictor(sam)
        self.device = config.device

        print(f"Model loaded successfully on {self.device}")

        self.loss_fn = LossFunction()

    def prepare_image(self, image_path: str):
        """准备图像和baseline"""
        print(f"\n{'='*70}")
        print(f"Preparing Image and Baseline")
        print(f"{'='*70}")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        print(f"Image loaded: {image_path}")
        print(f"Image shape: {image.shape}")

        # 设置提示点
        if self.config.point_coords is None:
            self.config.point_coords = np.array([[w // 2, int(h * 0.4)]])

        print(f"prompt point: {self.config.point_coords[0]}")

        # 获取benign样本的分割结果
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=self.config.point_coords,
            point_labels=np.array([self.config.point_label]),
            multimask_output=False,
        )
        baseline = {
            "image": image,
            "mask": masks[0],
            "score": scores[0],
            "logits": logits[0],
        }

        print(f" Baseline established:")
        print(f"  - Mask area: {masks[0].sum()} pixels")
        print(f"  - Quality score: {scores[0]:.4f}")

        return image, baseline

    def get_loss(self, logits, scores, loss_type: str):
        """根据类型计算损失"""
        if loss_type == "minimize_area":
            return self.loss_fn.minimize_area(logits, scores)
        elif loss_type == "minimize_score":
            return self.loss_fn.minimize_score(logits, scores)
        elif loss_type == "bce_loss":
            return self.loss_fn.bce_loss(logits, scores)
        elif loss_type == "combined":
            return self.loss_fn.combined_loss(
                logits, scores, self.config.combined_alpha, self.config.combined_beta
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """将numpy图像转换为tensor"""
        tensor = torch.from_numpy(image).float().to(self.device) / 255
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        return tensor

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """将tensor转为numpy图像"""
        image = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return image

    def predict_from_tensor(self, image_tensor: torch.Tensor):
        """
        从tensor预测并返回可微分的结果
        这是FGSM/PGD攻击的核心 - 需要梯度流动
        直接使用sam来保持梯度
        """
        # image_tensor shape: [3, H, W], 值范围 [0, 1]

        # SAM需要的预处理
        # 1. 调整到SAM的输入大小 (1024x1024)
        img_size = self.sam.image_encoder.img_size
        image_resized = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

        # 2.归一化
        pixel_mean = (
            torch.tensor([123.675, 116.28, 103.53], device=self.device).view(1, 3, 1, 1)
            / 255
        )
        pixel_std = (
            torch.tensor([58.395, 57.12, 57.375], device=self.device).view(1, 3, 1, 1)
            / 255
        )
        image_normalized = (image_resized - pixel_mean) / pixel_std

        #  3. 通过图像编码器
        with torch.set_grad_enabled(True):
            image_embedding = self.sam.image_encoder(image_normalized)

        # 4. 准备prompt
        # 将point坐标转换为SAM的输入格式
        coords = torch.tensor(
            self.config.point_coords, device=self.device, dtype=torch.float32
        )
        labels = torch.tensor(
            [self.config.point_label], device=self.device, dtype=torch.float32
        )

        # 缩放坐标到1024x1024
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        coords_scaled = coords.clone()
        coords_scaled[..., 0] = coords[..., 0] * (img_size / w)
        coords_scaled[..., 1] = coords[..., 1] * (img_size / h)

        # 5. 通过prompt编码器和mask解码器
        with torch.set_grad_enabled(True):
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(coords_scaled.unsqueeze(0), labels.unsqueeze(0)),
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

        # 6. 上采样到原始大小
        masks_upscaled = F.interpolate(
            low_res_masks, size=(h, w), mode="bilinear", align_corners=False
        )

        # 返回可微分的结果
        logits_tensor = masks_upscaled.squeeze(0)  # [1, H, W]
        scores_tensor = iou_predictions.squeeze(0)  # [1]

        # 同时返回numpy版本（用于显示）
        with torch.no_grad():
            masks_np = (
                (masks_upscaled.squeeze(0).sigmoid() > 0.5).cpu().numpy().astype(bool)
            )
            scores_np = iou_predictions.squeeze(0).cpu().numpy()

        return masks_np[0], scores_np[0], logits_tensor[0], scores_tensor[0]

    def fgsm_attack(self, image: np.ndarray, epsilon: float, loss_type: str):
        """
        FGSM攻击实现
        Fast Gradient Sign Method - 单步梯度符号攻击
        """

        # 转为tensor并计算梯度
        image_tensor = self.image_to_tensor(image)
        image_tensor.requires_grad = True

        # 前向传播
        masks, scores, logits, scores_tensor = self.predict_from_tensor(image_tensor)

        # 计算损失
        # 这里为什么要unsequeeze? 匹配Loss函数的期望输入维度
        loss = self.get_loss(logits.unsqueeze(0), scores_tensor, loss_type)

        # 反向传播
        loss.backward()

        # 生成扰动
        gradient = image_tensor.grad
        perturbation = epsilon * gradient.sign()

        # 生成对抗样本
        adv_image_tensor = image_tensor + perturbation
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)

        # 转回np格式的图片
        adv_image = self.tensor_to_image(adv_image_tensor)
        # 这里为什么要abs()?因为图像的有效值是[0,255],不包含负数. 为了生成一张能够被人眼观察的、代表扰动“位置”和“强度”的可视化图像。
        # 它不再区分扰动是增加像素值还是减小像素值（方向），而是统一显示所有被修改过的像素点（位置和幅度）。
        perturbation_np = self.tensor_to_image(perturbation.abs())

        # 在对抗样本上进行预测
        self.predictor.set_image(adv_image)
        adv_masks, adv_scores, adv_logits = self.predictor.predict(
            point_coords=self.config.point_coords,
            point_labels=np.array([self.config.point_label]),
            multimask_output=False,
        )

        return {
            "adv_image": adv_image,
            "adv_mask": adv_masks[0],
            "adv_score": adv_scores[0],
            "perturbation": perturbation_np,
            "loss_value": loss.item(),
        }

    def pgd_attack(
        self,
        image: np.ndarray,
        epsilon: float,
        alpha: float,
        steps: int,
        loss_type: str,
    ):
        """
        PGD攻击实现
        Projected Gradient Descent - 多步迭代优化攻击
        """
        # 转为tensor
        ori_image_tensor = self.image_to_tensor(image)

        # 初始化对抗样本
        # 为什么要clone().detach()?
        adv_image_tensor = ori_image_tensor.clone().detach()
        random_noise = torch.empty_like(adv_image_tensor).uniform_(-epsilon, epsilon)
        adv_image_tensor = adv_image_tensor + random_noise
        adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)

        loss_history = []

        # 迭代攻击
        for step in range(steps):
            adv_image_tensor.requires_grad = True

            # 前向传播
            mask, score, logits, score_tensor = self.predict_from_tensor(
                adv_image_tensor
            )

            # 计算损失
            loss = self.get_loss(logits.unsqueeze(0), score_tensor, loss_type)
            loss_history.append(loss.item())

            # 反向传播
            loss.backward()

            # 更新对抗样本
            gradient = adv_image_tensor.grad
            # 这里为什么要detach?
            adv_image_tensor = adv_image_tensor.detach() + alpha * gradient.sign()

            # 投影到epsilon球内
            perturbation = adv_image_tensor - ori_image_tensor
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            # 前面已经更新对抗样本了,为什么还需要这两行代码?
            adv_image_tensor = ori_image_tensor + perturbation
            adv_image_tensor = torch.clamp(adv_image_tensor, 0, 1)

            if (step + 1) % 5 == 0 or step == 0:
                print(
                    f"  Step {step+1}/{steps} | Loss: {loss.item():.6f} | "
                    f"Mask area: {mask.sum():.0f}"
                )

        # 转回numpy
        adv_image = self.tensor_to_image(adv_image_tensor)
        perturbation_np = self.tensor_to_image(
            (adv_image_tensor - ori_image_tensor).abs()
        )

        # 在对抗样本上预测
        adv_mask, adv_score, adv_logits_tensor, adv_scores_tensor = (
            self.predict_from_tensor(adv_image_tensor)
        )

        return {
            "adv_image": adv_image,
            "adv_mask": adv_mask,
            "adv_score": adv_score,
            "perturbation": perturbation_np,
            "loss_history": loss_history,
        }


# ==================== 评估器 ====================


class Evaluator:
    """评估指标计算和分析"""

    @staticmethod
    def calculate_metrics(
        baseline: dict, attack_result: dict, epsilon: float, method: str, loss_type: str
    ) -> dict:
        """计算所有评估指标"""
        clean_mask = baseline["mask"]
        adv_mask = attack_result["adv_mask"]

        # 主要指标
        # 为什么要进行sum?
        clean_area = clean_mask.sum()
        adv_area = adv_mask.sum()
        mask_area_ratio = adv_area / clean_area if clean_area > 0 else 0

        # 计算IoU 为什么用mask而不是area来计算?

        # 计算交集
        intersection = np.logical_and(clean_mask, adv_mask).sum()
        # 计算并集
        union = np.logical_or(clean_mask, adv_mask).sum()

        iou = intersection / union if union > 0 else 0

        # 质量分数下降
        clean_score = baseline["score"]
        adv_score = attack_result["adv_score"]
        score_drop = clean_score - adv_score

        # 扰动范数
        # 这里为什么 / 255 ?
        perturbation = attack_result["perturbation"].astype(float) / 255
        l_inf = np.abs(perturbation).max()
        l2 = np.sqrt((perturbation**2).sum())

        # 攻击成功判定
        attack_success = (mask_area_ratio < 0.1) or (iou < 0.3)

        metrics = {
            "method": method,
            "loss_type": loss_type,
            "epsilon": epsilon,
            "clean_area": int(clean_area),
            "adv_area": int(adv_area),
            "mask_area_ratio": float(mask_area_ratio),
            "iou": float(iou),
            "clean_score": float(clean_score),
            "adv_score": float(adv_score),
            "score_drop": float(score_drop),
            "l_inf": float(l_inf),
            "l2": float(l2),
            "attack_success": bool(attack_success),
        }

        return metrics


# ==================== 可视化 ====================
class Visualizer:

    @staticmethod
    def visualize_attack_result(
        baseline: dict, attack_result: dict, metrics: dict, save_path: str
    ):
        """可视化单个攻击结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 第一行：图像对比
        axes[0, 0].imshow(baseline["image"])
        axes[0, 0].set_title("Clean Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(attack_result["adv_image"])
        axes[0, 1].set_title("Adversarial Image", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        perturbation_vis = attack_result["perturbation"]
        axes[0, 2].imshow(perturbation_vis, cmap="hot")
        axes[0, 2].set_title(
            f'Perturbation\nL∞={metrics["l_inf"]:.4f}', fontsize=12, fontweight="bold"
        )
        axes[0, 2].axis("off")

        # 第二行：Mask对比
        axes[1, 0].imshow(baseline["image"])
        axes[1, 0].imshow(baseline["mask"], alpha=0.5, cmap="jet")
        axes[1, 0].set_title(
            f'Clean Mask\nArea: {metrics["clean_area"]} | Score: {metrics["clean_score"]:.3f}',
            fontsize=11,
        )
        axes[1, 0].axis("off")

        axes[1, 1].imshow(attack_result["adv_image"])
        axes[1, 1].imshow(attack_result["adv_mask"], alpha=0.5, cmap="jet")
        axes[1, 1].set_title(
            f'Adversarial Mask\nArea: {metrics["adv_area"]} | Score: {metrics["adv_score"]:.3f}',
            fontsize=11,
        )
        axes[1, 1].axis("off")

        # 指标总结
        axes[1, 2].axis("off")
        success_str = "SUCCESS" if metrics["attack_success"] else "FAILED"
        success_color = "green" if metrics["attack_success"] else "red"

        metrics_text = f"""
Method: {metrics['method']}
Loss: {metrics['loss_type']}
Epsilon: {metrics['epsilon']:.4f}

Mask Area Ratio: {metrics['mask_area_ratio']:.4f}
IoU: {metrics['iou']:.4f}
Score Drop: {metrics['score_drop']:.4f}

L∞ Norm: {metrics['l_inf']:.4f}
L2 Norm: {metrics['l2']:.2f}
        """

        axes[1, 2].text(
            0.1,
            0.6,
            metrics_text,
            fontsize=11,
            verticalalignment="center",
            family="monospace",
        )
        axes[1, 2].text(
            0.5,
            0.15,
            success_str,
            fontsize=14,
            verticalalignment="center",
            ha="center",
            fontweight="bold",
            color=success_color,
        )

        plt.suptitle(
            f'{metrics["method"]} Attack - {metrics["loss_type"]}',
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ==================== 实验运行器 ====================


class ExperimentRunner:
    """完整实验流程管理"""

    def __init__(self, config: AttackConfig):
        self.config = config
        self.attacker = SAMAttacker(config)
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        self.all_results = []

    def run_all_experiments(self):
        """运行所有实验配置"""
        print(f"\n{'='*70}")
        print("SAM ADVERSARIAL ATTACK RESEARCH")
        print("Objective: Make SAM 'Segment Nothing'")
        print(f"{'='*70}")

        # 准备图像和baseline
        image, baseline = self.attacker.prepare_image(self.config.image_path)
        self.save_baseline(baseline)

        # 实验1: FGSM攻击
        print(f"\n{'='*70}")
        print("EXPERIMENT 1: FGSM ATTACKS")
        print(f"{'='*70}")
        self.run_fgsm_experiments(image, baseline)

        # 实验2: PGD攻击
        print(f"\n{'='*70}")
        print("EXPERIMENT 2: PGD ATTACKS")
        print(f"{'='*70}")
        self.run_pgd_experiments(image, baseline)

        # 保存和分析结果
        self.save_results()
        self.analyze_results()

        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"{'='*70}")
        print(f"Total configurations tested: {len(self.all_results)}")
        print(f"Results saved in: {self.config.results_dir}")
        print(f"{'='*70}")

    def run_fgsm_experiments(self, image, baseline):
        """运行所有FGSM实验"""
        total = len(self.config.loss_types) * len(self.config.fgsm_epsilon)
        current = 0

        for loss_type in self.config.loss_types:
            for epsilon in self.config.fgsm_epsilon:
                current += 1
                print(
                    f"\n[{current}/{total}] FGSM | Loss: {loss_type} | Epsilon: {epsilon:.4f}"
                )
                print("-" * 70)

                # 执行攻击
                attack_result = self.attacker.fgsm_attack(image, epsilon, loss_type)

                # 评估
                metrics = self.evaluator.calculate_metrics(
                    baseline, attack_result, epsilon, "FGSM", loss_type
                )
                self.all_results.append(metrics)

                # 打印结果
                self._print_metrics(metrics)

                # 可视化
                save_name = f"fgsm_{loss_type}_eps{int(epsilon*255)}.png"
                save_path = os.path.join(self.config.figures_dir, save_name)
                self.visualizer.visualize_attack_result(
                    baseline, attack_result, metrics, save_path
                )
                print(f"Saved: {save_name}")

    def run_pgd_experiments(self, image, baseline):
        """运行所有PGD实验"""
        total = len(self.config.loss_types) * len(self.config.pgd_steps_list)
        current = 0

        for loss_type in self.config.loss_types:
            for steps in self.config.pgd_steps_list:
                current += 1
                print(f"\n[{current}/{total}] PGD | Loss: {loss_type} | Steps: {steps}")
                print("-" * 70)

                # 执行攻击
                attack_result = self.attacker.pgd_attack(
                    image,
                    self.config.pgd_epsilon,
                    self.config.pgd_alpha,
                    steps,
                    loss_type,
                )

                # 评估
                metrics = self.evaluator.calculate_metrics(
                    baseline,
                    attack_result,
                    self.config.pgd_epsilon,
                    f"PGD-{steps}",
                    loss_type,
                )
                self.all_results.append(metrics)

                # 打印结果
                self._print_metrics(metrics)

                # 可视化
                save_name = f"pgd_{loss_type}_steps{steps}.png"
                save_path = os.path.join(self.config.figures_dir, save_name)
                self.visualizer.visualize_attack_result(
                    baseline, attack_result, metrics, save_path
                )
                print(f"Saved: {save_name}")

    def _print_metrics(self, metrics):
        """打印关键指标"""
        success = "SUCCESS" if metrics["attack_success"] else "FAILED"
        print(
            f"Results: MAR={metrics['mask_area_ratio']:.4f} | "
            f"IoU={metrics['iou']:.4f} | "
            f"ScoreDrop={metrics['score_drop']:.4f} | {success}"
        )

    def save_baseline(self, baseline):
        """保存baseline结果"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(baseline["image"])
        point = self.config.point_coords[0]
        axes[0].scatter(
            point[0],
            point[1],
            c="green",
            s=300,
            marker="*",
            edgecolors="white",
            linewidths=2,
        )
        axes[0].set_title(
            "Clean Image with Prompt Point", fontsize=12, fontweight="bold"
        )
        axes[0].axis("off")

        axes[1].imshow(baseline["image"])
        axes[1].imshow(baseline["mask"], alpha=0.5, cmap="jet")
        axes[1].set_title(
            f'Baseline Segmentation\nArea: {baseline["mask"].sum():.0f} pixels | Score: {baseline["score"]:.4f}',
            fontsize=12,
            fontweight="bold",
        )
        axes[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.config.figures_dir, "baseline.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Baseline saved: {save_path}")

    def save_results(self):
        """保存实验结果"""
        df = pd.DataFrame(self.all_results)

        # 保存CSV
        csv_path = os.path.join(self.config.logs_dir, "results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults CSV saved: {csv_path}")

        # 保存JSON
        json_path = os.path.join(self.config.logs_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(self.all_results, f, indent=2)
        print(f"Results JSON saved: {json_path}")

    def analyze_results(self):
        """分析实验结果并识别最佳方法"""
        df = pd.DataFrame(self.all_results)

        print(f"\n{'='*70}")
        print("RESULTS ANALYSIS")
        print(f"{'='*70}")

        # 1. 按损失函数分组
        print("\n1. Performance by Loss Function:")
        print("-" * 70)
        loss_stats = (
            df.groupby("loss_type")
            .agg(
                {
                    "mask_area_ratio": ["mean", "min"],
                    "iou": ["mean", "min"],
                    "score_drop": ["mean", "max"],
                    "attack_success": "mean",
                }
            )
            .round(4)
        )
        print(loss_stats)

        # 2. 按方法分组
        print("\n2. Performance by Attack Method:")
        print("-" * 70)
        method_stats = (
            df.groupby("method")
            .agg(
                {
                    "mask_area_ratio": ["mean", "min"],
                    "iou": ["mean", "min"],
                    "attack_success": "mean",
                }
            )
            .round(4)
        )
        print(method_stats)

        # 3. Top 5配置
        print("\n3. Top 5 Configurations by Mask Area Ratio (Lower is Better):")
        print("-" * 70)
        top5 = df.nsmallest(5, "mask_area_ratio")[
            [
                "method",
                "loss_type",
                "mask_area_ratio",
                "iou",
                "score_drop",
                "attack_success",
            ]
        ]
        print(top5.to_string(index=False))

        # 4. 攻击成功率
        print("\n4. Attack Success Rate by Configuration:")
        print("-" * 70)
        success_rate = df.groupby(["method", "loss_type"])["attack_success"].agg(
            ["mean", "sum", "count"]
        )
        success_rate.columns = ["Success Rate", "Successes", "Total"]
        success_rate["Success Rate"] = (success_rate["Success Rate"] * 100).round(2)
        print(success_rate)

        # 5. 最佳方法
        print(f"\n{'='*70}")
        print("★ BEST ATTACK METHOD ★")
        print(f"{'='*70}")
        best_row = df.loc[df["mask_area_ratio"].idxmin()]
        print(f"Method:           {best_row['method']}")
        print(f"Loss Function:    {best_row['loss_type']}")
        print(f"Epsilon:          {best_row['epsilon']:.4f}")
        print(f"─" * 70)
        print(f"Mask Area Ratio:  {best_row['mask_area_ratio']:.4f}  (Target: < 0.1)")
        print(f"IoU:              {best_row['iou']:.4f}  (Target: < 0.3)")
        print(f"Score Drop:       {best_row['score_drop']:.4f}  (Target: > 0.5)")
        print(f"Attack Success:   {best_row['attack_success']}")
        print(f"{'='*70}")

        # 生成对比图
        self.create_comparison_plots(df)

    def create_comparison_plots(self, df):
        """创建对比可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: MAR by Loss Function
        loss_mar = df.groupby("loss_type")["mask_area_ratio"].mean().sort_values()
        colors = ["green" if x < 0.1 else "orange" for x in loss_mar.values]
        axes[0, 0].barh(range(len(loss_mar)), loss_mar.values, color=colors)
        axes[0, 0].set_yticks(range(len(loss_mar)))
        axes[0, 0].set_yticklabels(loss_mar.index)
        axes[0, 0].axvline(0.1, color="red", linestyle="--", label="Success Threshold")
        axes[0, 0].set_xlabel("Mask Area Ratio")
        axes[0, 0].set_title("Average MAR by Loss Function\n(Lower is Better)")
        axes[0, 0].legend()
        axes[0, 0].grid(axis="x", alpha=0.3)

        # 图2: IoU by Loss Function
        loss_iou = df.groupby("loss_type")["iou"].mean().sort_values()
        colors = ["green" if x < 0.3 else "orange" for x in loss_iou.values]
        axes[0, 1].barh(range(len(loss_iou)), loss_iou.values, color=colors)
        axes[0, 1].set_yticks(range(len(loss_iou)))
        axes[0, 1].set_yticklabels(loss_iou.index)
        axes[0, 1].axvline(0.3, color="red", linestyle="--", label="Success Threshold")
        axes[0, 1].set_xlabel("IoU")
        axes[0, 1].set_title("Average IoU by Loss Function\n(Lower is Better)")
        axes[0, 1].legend()
        axes[0, 1].grid(axis="x", alpha=0.3)

        # 图3: Success Rate by Loss Function
        success_by_loss = df.groupby("loss_type")["attack_success"].mean() * 100
        success_by_loss = success_by_loss.sort_values(ascending=False)
        colors = ["green" if x >= 50 else "red" for x in success_by_loss.values]
        axes[1, 0].barh(
            range(len(success_by_loss)), success_by_loss.values, color=colors
        )
        axes[1, 0].set_yticks(range(len(success_by_loss)))
        axes[1, 0].set_yticklabels(success_by_loss.index)
        axes[1, 0].set_xlabel("Success Rate (%)")
        axes[1, 0].set_title("Attack Success Rate by Loss Function")
        axes[1, 0].set_xlim([0, 105])
        axes[1, 0].grid(axis="x", alpha=0.3)

        # 图4: Method Comparison
        method_mar = df.groupby("method")["mask_area_ratio"].mean().sort_values()
        axes[1, 1].bar(range(len(method_mar)), method_mar.values, color="steelblue")
        axes[1, 1].set_xticks(range(len(method_mar)))
        axes[1, 1].set_xticklabels(method_mar.index, rotation=45)
        axes[1, 1].axhline(0.1, color="red", linestyle="--", label="Success Threshold")
        axes[1, 1].set_ylabel("Mask Area Ratio")
        axes[1, 1].set_title("Average MAR by Attack Method")
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", alpha=0.3)

        plt.suptitle(
            "Adversarial Attack Performance Analysis", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        save_path = os.path.join(self.config.figures_dir, "comparison_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nComparison analysis saved: {save_path}")


# ==================== 主函数 ====================
def main():
    """运行完整实验"""
    print("\n" + "=" * 70)
    print("SAM ADVERSARIAL ATTACK RESEARCH")
    print("Making SAM 'Segment Nothing'")
    print("=" * 70)

    # 创建配置
    config = AttackConfig()

    # 创建实验运行器
    runner = ExperimentRunner(config)

    # 运行所有实验
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
