import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry,SamPredictor

# ==================== 1. 加载模型和图像 ====================
print("Step 1: 加载SAM模型...")
sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)
predictor = SamPredictor(sam)
print(f"模型已加载到 {device}")

# 读取图像
print("\nStep 2: 读取并编码图像...")
image = cv2.imread('cat.png')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
print(f"图像尺寸: {image.shape}")

# 编码图像（这一步会计算embeddings，对同一图像只需做一次）
predictor.set_image(image)
print("图像编码完成！")

# ==================== 3. 示例2：多点提示（前景+背景）====================
print("\n" + "="*50)
print("示例2: 使用多点提示（前景+背景）")
print("="*50)

# 前景点（猫的面部和耳朵）和背景点（蓝色背景区域）
h, w = image.shape[:2]
input_points = np.array([
    [w//2,int(h*0.4)],
    [int(w*0.4),int(h*0.3)],
    [50,50],
    [w-50,50]
])
input_labels = np.array([1,1,0,0])

masks,scores,logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)

best_mask_idx = np.argmax(scores)
best_mask = masks[best_mask_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image)
axes[0].scatter(input_points[input_labels==1, 0], input_points[input_labels==1, 1], 
                c='green', s=150, marker='*', label='foreground')
axes[0].scatter(input_points[input_labels==0, 0], input_points[input_labels==0, 1], 
                c='red', s=150, marker='x', label='background')
axes[0].legend()
axes[0].set_title('prompt points')
axes[0].axis('off')

axes[1].imshow(image)
axes[1].imshow(best_mask, alpha=0.5, cmap='jet')
axes[1].set_title(f'best segment result(score: {scores[best_mask_idx]:.3f})')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('result_2_multipoint_prompt.png', dpi=150, bbox_inches='tight')
print("结果已保存到: result_2_multipoint_prompt.png")
plt.close()