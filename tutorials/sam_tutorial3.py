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

h, w = image.shape[:2]

# ==================== 4. 示例3：边界框提示 ====================
print("\n" + "="*50)
print("示例3: 使用边界框提示")
print("="*50)

# 定义包含猫头的边界框 [x_min, y_min, x_max, y_max]
# 猫头大致在中央，占据图片的中间60%区域

input_box = np.array([
    int(w * 0.2),      # x_min: 左边界
    int(h * 0.1),      # y_min: 上边界（猫耳朵上方）
    int(w * 0.8),      # x_max: 右边界
    int(h * 0.7)       # y_max: 下边界（猫下巴下方）
])

masks, scores, logits = predictor.predict(
    box=input_box,
    multimask_output=False  # 框提示通常只需要一个mask
)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制边界框
axes[0].imshow(image)
rect = plt.Rectangle((input_box[0], input_box[1]), 
                      input_box[2]-input_box[0], 
                      input_box[3]-input_box[1],
                      fill=False, edgecolor='red', linewidth=3)
axes[0].add_patch(rect)
axes[0].set_title('input box')
axes[0].axis('off')

axes[1].imshow(image)
axes[1].imshow(masks[0], alpha=0.5, cmap='jet')
axes[1].set_title(f'segment result(score: {scores[0]:.3f})')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('result_3_box_prompt.png', dpi=150, bbox_inches='tight')
print("结果已保存到: result_3_box_prompt.png")
plt.close()