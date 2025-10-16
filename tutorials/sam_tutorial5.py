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

# ==================== 6. 提取分割对象 ====================
print("\n" + "="*50)
print("示例5: 提取分割出的对象")
print("="*50)

# 使用最后一个mask
mask = masks[0]

# 创建透明背景版本
segmented = image.copy()
segmented[~mask] = [255, 255, 255]  # 背景变白

rgba = np.zeros((image.shape[0],image.shape[1],4),dtype=np.uint8)
rgba[:,:,:3] = image
rgba[:,:,3] = (mask * 255).astype(np.uint8) 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(segmented)
axes[0].set_title('white background')
axes[0].axis('off')

axes[1].imshow(rgba)
axes[1].set_title('transparent background')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('result_5_extracted_object.png', dpi=150, bbox_inches='tight')
print("结果已保存到: result_5_extracted_object.png")
plt.close()

# 保存透明背景PNG
cv2.imwrite('cat_segmented.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
print("透明背景图已保存到: cat_segmented.png")
