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

# ==================== 2. 示例1：点提示分割 ====================
print("\n" + "="*50)
print("示例1: 使用点提示分割猫")
print("="*50)

# 在猫的鼻子/面部选择一个前景点


h,w = image.shape[:2]
input_point = np.array([[w//2,int(h*0.4)]]) # 猫的面部中心
input_label = np.array([1])  # 1表示前景


masks,scores,logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

print(f"生成了 {len(masks)} 个候选mask")
print(f"质量分数: {scores}")

# 可视化结果
fig,axes = plt.subplots(1,4,figsize=(16,4))
axes[0].imshow(image)
axes[0].scatter(input_point[:,0],input_point[:,1],c='red',s=100,marker='*')
axes[0].axis('off')

for i,(mask,score) in enumerate(zip(masks,scores)):
    axes[i+1].imshow(image)
    axes[i+1].imshow(mask, alpha=0.5, cmap='jet')
    axes[i+1].set_title(f'Mask {i+1} (score: {score:.3f})')
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('result_1_point_prompt.png', dpi=150, bbox_inches='tight')
print("结果已保存到: result_1_point_prompt.png")
plt.close()