import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm

safe_tt = "hidden_states/llava-1.5-7b-ifeval-text.npy"
safe_tt_2 = "hidden_states/llava-1.5-7b-alpaca-text.npy"
unsafe_tt = "hidden_states/llava-1.5-7b-mmsafetybench_local_text-text.npy"
safe_vl = "hidden_states/llava-1.5-7b-mmvet-default.npy"
unsafe_vl = "hidden_states/llava-1.5-7b-mmsafetybench_local_sd-default.npy"

safe_tt_with_vl = "hidden_states/llava-1.5-7b-ifeval-nature.npy"
safe_tt_with_vl_2 = "hidden_states/llava-1.5-7b-alpaca-nature.npy"

llm_tt = "hidden_states/llm-vicuna7b-ifeval-text.npy"
llm_tt_2 = "hidden_states/llm-vicuna7b-alpaca-text.npy"
llm_gemma_tt = "hidden_states/llm-gemma-ifeval-text.npy"
llm_gemma_tt_2 = "hidden_states/llm-gemma-alpaca-text.npy"
llm_gemma_unsafe = "hidden_states/llm-gemma-mmsafetybench_local_text-text.npy"
llm_llama3_tt = "hidden_states/llm-llama3-ifeval-text.npy"
llm_llama3_tt_2 = "hidden_states/llm-llama3-alpaca-text.npy"
llm_llama3_unsafe = "hidden_states/llm-llama3-mmsafetybench_local_text-text.npy"
llm_llama2_tt = "hidden_states/llm-llama2-ifeval-text.npy"
llm_llama2_tt_2 = "hidden_states/llm-llama2-alpaca-text.npy"
llm_llama2_unsafe = "hidden_states/llm-llama2-mmsafetybench_local_text-text.npy"
llm_phi3_tt = "hidden_states/llm-phi3-mini-4k-ifeval-text.npy"
llm_phi3_tt_2 = "hidden_states/llm-phi3-mini-4k-alpaca-text.npy"
llm_phi3_unsafe = "hidden_states/llm-phi3-mini-4k-mmsafetybench_local_text-text.npy"



# --------------------------
# 1. 加载数据
# --------------------------
# 替换为你的npy文件路径
file1_path = llm_llama2_tt
file2_path = llm_llama2_unsafe

# 加载数据
try:
    data1 = np.load(file1_path)
    data2 = np.load(file2_path)
    print(f"成功加载数据：")
    print(f"数据集1形状: {data1.shape} (样本数, 层数, 隐藏维度)")
    print(f"数据集2形状: {data2.shape} (样本数, 层数, 隐藏维度)")
except FileNotFoundError as e:
    print(f"文件加载失败: {e}")
    exit(1)
except Exception as e:
    print(f"数据加载错误: {e}")
    exit(1)

# 检查层数和隐藏维度是否一致
if data1.shape[1] != data2.shape[1] or data1.shape[2] != data2.shape[2]:
    raise ValueError("两个数据集的层数或隐藏维度不一致，无法比较！")

# --------------------------
# 2. 计算各层平均隐藏状态
# --------------------------
# 对样本维度取平均 (size, layer, dim) -> (layer, dim)
mean_data1 = np.mean(data1, axis=0)  # 数据集1各层平均
mean_data2 = np.mean(data2, axis=0)  # 数据集2各层平均

print(f"平均后形状: {mean_data1.shape} (层数, 隐藏维度)")

# --------------------------
# 3. 计算各层隐藏状态的绝对差值
# --------------------------
# 计算两层平均隐藏状态的绝对值差，然后可以对隐藏维度取平均或求和（根据需求选择）
# 方式1：对隐藏维度取平均（反映整体平均差异）
# layer_diff = np.mean(np.abs(mean_data1 - mean_data2), axis=1)
layer_diff = np.array([
    norm(mean_data1[layer] - mean_data2[layer])  # 欧氏距离 = L2范数
    for layer in range(mean_data1.shape[0])
])

# 方式2：计算两层平均隐藏状态的余弦相似度
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)  # 点积
    norm_vec1 = np.linalg.norm(vec1)  # 向量1的模长
    norm_vec2 = np.linalg.norm(vec2)  # 向量2的模长
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # 避免除以零
    return dot_product / (norm_vec1 * norm_vec2)

# 对每一层计算余弦相似度
layer_similarity = np.array([
    cosine_similarity(mean_data1[layer], mean_data2[layer]) 
    for layer in range(mean_data1.shape[0])
])

# 获取层数
layer_num = mean_data1.shape[0]
layers = np.arange(0, layer_num)

# --------------------------
# 3. 计算角度差（核心新增逻辑）
# --------------------------
def angle_between_vectors(v1, v2):
    """计算两个向量的夹角（弧度制）"""
    dot = np.dot(v1, v2)
    norm_v1 = norm(v1)
    norm_v2 = norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = dot / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差导致超出范围
    theta_rad = np.arccos(cos_theta)  # 弧度制（范围：[0, π]）
    theta_deg = np.rad2deg(theta_rad)  # 角度制（范围：[0, 180] 度）
    
    return theta_deg

def compute_average_angle_diff(data1, data2, num_samples=1000):
    """
    随机抽样计算角度差：
    1. 计算 data1 内部样本对的角度（N-N 对）
    2. 计算 data1 与 data2 交错样本对的角度（N-M 对）
    3. 逐对计算角度差并取平均
    """
    size1, layer_num, dim = data1.shape
    size2 = data2.shape[0]
    angle_diff_per_layer = np.zeros(layer_num)  # 每层的角度差结果

    # 随机种子固定（可选，保证可复现）
    np.random.seed(42)  

    for layer in range(layer_num):
        n_n_angles = []  # N-N 对的角度
        n_m_angles = []  # N-M 对的角度

        for _ in range(num_samples):
            # 抽样 N-N 对：data1 内部两个样本
            p, p_prime = np.random.choice(size1, 2, replace=False)
            v1 = data1[p, layer]
            v2 = data1[p_prime, layer]
            n_n_angles.append(angle_between_vectors(v1, v2))

            # 抽样 N-M 对：data1 和 data2 各一个样本
            # p = np.random.choice(size1)
            q = np.random.choice(size2)
            v1 = data1[p, layer]
            v2 = data2[q, layer]
            n_m_angles.append(angle_between_vectors(v1, v2))

        # 计算角度差的平均值（N-M 角度 - N-N 角度）
        n_n_avg = np.mean(n_n_angles)
        n_m_avg = np.mean(n_m_angles)
        angle_diff_per_layer[layer] = n_m_avg - n_n_avg  # 角度差（N-M - N-N）

    return angle_diff_per_layer

# 执行角度差计算（可调整 num_samples 控制抽样次数）
layer_angle_diff = compute_average_angle_diff(data1, data2, num_samples=5000)

def compute_average_distance_diff(data1, data2, num_samples=1000):
    size1, layer_num, dim = data1.shape
    size2 = data2.shape[0]
    distance_diff_per_layer = np.zeros(layer_num)  # 每层的角度差结果

    # 随机种子固定（可选，保证可复现）
    np.random.seed(42)  

    for layer in range(layer_num):
        n_n_distance = []  # N-N 对的角度
        n_m_distance = []  # N-M 对的角度

        for _ in range(num_samples):
            # 抽样 N-N 对：data1 内部两个样本
            p, p_prime = np.random.choice(size1, 2, replace=False)
            v1 = data1[p, layer]
            v2 = data1[p_prime, layer]
            n_n_distance.append(norm(v1 - v2))  # 欧氏距离

            # 抽样 N-M 对：data1 和 data2 各一个样本
            # p = np.random.choice(size1)
            q = np.random.choice(size2)
            v1 = data1[p, layer]
            v2 = data2[q, layer]
            n_m_distance.append(norm(v1 - v2))

        n_n_avg = np.mean(n_n_distance)
        n_m_avg = np.mean(n_m_distance)
        distance_diff_per_layer[layer] = n_m_avg - n_n_avg 

    return distance_diff_per_layer

layer_distance_diff = compute_average_distance_diff(data1, data2, num_samples=5000)

# # --------------------------
# # 4. 可视化结果
# # --------------------------
# plt.style.use('seaborn-v0_8-notebook')  # 设置绘图风格

# plt.figure(figsize=(10, 6))
# plt.plot(layers, layer_diff, marker='o', linestyle='-', color='#3498db', 
#          markersize=8, linewidth=2, markerfacecolor='#e74c3c')

# # 添加标题和标签
# plt.title('Layer-wise Average dist', fontsize=15, pad=20)
# plt.xlabel('Layer', fontsize=12, labelpad=10)
# plt.ylabel('Dist', fontsize=12, labelpad=10)

# # 设置坐标轴范围和刻度
# plt.xlim(0, layer_num-1)
# plt.xticks(layers, fontsize=10)
# plt.ylim(0, max(layer_diff) * 1.1)  # 预留10%的顶部空间

# # 添加网格线
# plt.grid(alpha=0.3, linestyle='--')

# # 添加数值标签（可选）
# for i, value in enumerate(layer_diff):
#     plt.text(layers[i], value + max(layer_diff)*0.02, 
#              f'{value:.4f}', ha='center', fontsize=9)

# # 调整布局
# plt.tight_layout()

# # 保存图像（可选）
# output_dir = "visualization_results"
# os.makedirs(output_dir, exist_ok=True)
# plt.savefig(os.path.join(output_dir, 'layer_hidden_state_dist.png'), dpi=300, bbox_inches='tight')

# # 显示图像
# plt.show()

# --------------------------
# 5. 绘图：三图分栏（距离、余弦相似度、角度差）
# --------------------------
plt.style.use('seaborn-v0_8-notebook')  

# 创建 3 行 1 列子图（距离、余弦相似度、角度差）
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)  
fig.subplots_adjust(hspace=0.3)  

# --------------------------
# 子图 1：距离（原逻辑）
# --------------------------
ax1.plot(layers, layer_diff, marker='o', linestyle='-', color='#3498db', 
         markersize=8, linewidth=2, markerfacecolor='#e74c3c')
ax1.set_title('Layer-wise Average Distance', fontsize=15, pad=20)
ax1.set_ylabel('Distance', fontsize=12, labelpad=10)
ax1.set_ylim(0, max(layer_diff) * 1.1)  
ax1.grid(alpha=0.3, linestyle='--')

for i, value in enumerate(layer_diff):
    ax1.text(layers[i], value + max(layer_diff)*0.02, 
             f'{value:.4f}', ha='center', fontsize=9)

# --------------------------
# 子图 2：余弦相似度（原逻辑）
# --------------------------
ax2.plot(layers, layer_similarity, marker='s', linestyle='-', color='#2ecc71', 
         markersize=8, linewidth=2, markerfacecolor='#f39c12')
ax2.set_title('Layer-wise Cosine Similarity', fontsize=15, pad=20)
ax2.set_ylabel('Cosine Similarity', fontsize=12, labelpad=10)
ax2.set_ylim(-1.1, 1.1)  
ax2.grid(alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)  

for i, value in enumerate(layer_similarity):
    y_offset = 0.05 if value >= 0 else -0.1
    ax2.text(layers[i], value + y_offset, 
             f'{value:.4f}', ha='center', fontsize=9)

# --------------------------
# 子图 3：角度差（新增）
# --------------------------
ax3.plot(layers, layer_angle_diff, marker='^', linestyle='-', color='#f39c12', 
         markersize=8, linewidth=2, markerfacecolor='#3498db')
ax3.set_title('Layer-wise Average Angle Difference', fontsize=15, pad=20)
ax3.set_xlabel('Layer', fontsize=12, labelpad=10)  # 共享x轴，只在最后标xlabel
ax3.set_ylabel('Angle Diff (rad)', fontsize=12, labelpad=10)
ax3.set_ylim(0, max(layer_angle_diff) * 1.1)
ax3.grid(alpha=0.3, linestyle='--')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)  

for i, value in enumerate(layer_angle_diff):
    y_offset = 0.01 if value >= 0 else -0.02
    ax3.text(layers[i], value + y_offset, 
             f'{value:.4f}', ha='center', fontsize=9)
    
# --------------------------
# 子图 4：距离差
# --------------------------
ax4.plot(layers, layer_distance_diff, marker='D', linestyle='-', color='#9b59b6', 
         markersize=8, linewidth=2, markerfacecolor='#e67e22')
ax4.set_title('Layer-wise Average Distance Difference', fontsize=15, pad=20)
ax4.set_xlabel('Layer', fontsize=12, labelpad=10)  # 若为最后一个子图，保留xlabel
ax4.set_ylabel('Distance Diff', fontsize=12, labelpad=10)
ax4.set_ylim(0, max(layer_distance_diff) * 1.1)  # 距离差非负，从0开始
ax4.grid(alpha=0.3, linestyle='--')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)  # 参考零线 

for i, value in enumerate(layer_distance_diff):
    y_offset = 0.01 if value >= 0 else -0.02
    ax4.text(layers[i], value + y_offset, 
             f'{value:.4f}', ha='center', fontsize=9)

# --------------------------
# 保存与显示
# --------------------------
output_dir = "visualization_results"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'llm_llama2_tt_unsafe.png'), dpi=300, bbox_inches='tight')
plt.show()
