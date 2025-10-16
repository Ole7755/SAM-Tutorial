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
# ==================== 5. 示例4：组合提示（框+点）====================
print("\n" + "="*50)
print("示例4: 组合提示（边界框 + 点）")
print("="*50)

# 使用相同的框，但添加猫鼻子的精确前景点
input_box = np.array([
    int(w * 0.2),      # x_min: 左边界
    int(h * 0.1),      # y_min: 上边界（猫耳朵上方）
    int(w * 0.8),      # x_max: 右边界
    int(h * 0.7)       # y_max: 下边界（猫下巴下方）
])
input_point = np.array([[w//2, int(h*0.4)]])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False
)


# 可视化
plt.figure(figsize=(8, 6))
plt.imshow(image)
rect = plt.Rectangle((input_box[0], input_box[1]), 
                      input_box[2]-input_box[0], 
                      input_box[3]-input_box[1],
                      fill=False, edgecolor='red', linewidth=2, label='box')
plt.gca().add_patch(rect)
plt.scatter(input_point[:, 0], input_point[:, 1], 
            c='green', s=200, marker='*', label='foreground')
plt.imshow(masks[0], alpha=0.5, cmap='jet')
plt.title(f'combination prompt result(score: {scores[0]:.3f})')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.savefig('result_4_combined_prompt.png', dpi=150, bbox_inches='tight')
print("结果已保存到: result_4_combined_prompt.png")
plt.close()