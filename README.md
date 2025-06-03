# braincog-cifar10-demo
Brain-Cog 論文報告與程式實作：以 CifarConvNet 為例

> 論文來源：[BrainCog: A Spiking Neural Network based Brain-inspired Cognitive Intelligence Framework](https://arxiv.org/abs/2303.17469)  
> 原始碼倉庫：[BrainCog-X/Brain-Cog](https://github.com/BrainCog-X/Brain-Cog)

---

##  一、研究背景與動機

- 傳統神經網絡（ANN）在能耗與記憶表現上存在限制。  
- 脈衝神經網絡（SNN）模擬生物神經元，可實現更節能、具時間記憶特性的計算。  
- 論文提出 Spiking World Model（Spiking-WM），並引入多隔室神經元（MCN）設計，有效提升長序列處理能力。  
- `CifarConvNet` 是 Brain-Cog 框架中針對影像分類任務的代表模型。  

---

##  二、CifarConvNet 結構與設計邏輯

###  模型架構（簡略）

| Layer     | Type     | Size       | Activation |
|-----------|----------|------------|------------|
| Input     | Image    | 32×32×3    | –          |
| Conv2d_1  | Conv     | 256@3×3    | LIFNode    |
| Conv2d_2  | Conv     | 256@3×3    | LIFNode    |
| AvgPool   | Pooling  | 2×2        | –          |
| Conv2d_3  | Conv     | 256@3×3    | LIFNode    |
| Conv2d_4  | Conv     | 256@3×3    | LIFNode    |
| AvgPool   | Pooling  | 2×2        | –          |
| Conv2d_5  | Conv     | 512@3×3    | LIFNode    |
| Conv2d_6  | Conv     | 512@3×3    | LIFNode    |
| AvgPool   | Pooling  | 2×2        | –          |
| Conv2d_7  | Conv     | 1024@3×3   | LIFNode    |
| Conv2d_8  | Conv     | 1024@3×3   | LIFNode    |
| AdaptiveAvgPool | Pooling | 1×1    | –          |
| Flatten   | –        | –          | –          |
| FC        | Linear   | 1024→10    | Identity   |

> 每一個卷積層後都接 LIFNode（QGateGrad 作為 surrogate gradient），最後再接一層線性層輸出 10 個類別分數。

###  神經元設計：LIFNode

- Leaky Integrate-and-Fire 模型（LIF）：
  - 具備膜電位積累、洩漏與發放機制。  
  - 當膜電位超過閾值時發放一個 spike，然後重置膜電位。  
- 使用 surrogate gradient（例如 QGateGrad）解決脈衝發放時不可微問題，才能執行反向傳播訓練。  
- 輸入的時間步數由 `--step 8` 控制，即多個時間步上累積脈衝訊號。  

---

##  三、MCN 多隔室神經元：設計思路

| 區塊         | 功能說明                         |
|--------------|----------------------------------|
| 基底樹突 (Basal) | 接收外部輸入信號                   |
| 頂端樹突 (Apical) | 調節輸入訊號貢獻權重               |
| 軀體 (Soma)     | 整合樹突電位並決定是否發放脈衝         |

- **非線性門控整合**：頂端樹突的電位 \(V_a[t]\) 透過 sigmoid 函數控制基底樹突 \(V_b[t]\) 與 soma 膜電位的資訊貢獻。  
- **動態方程**（示意）：
  \[
    U[t] = \sigma(V_a[t]) \cdot V_b[t] + \text{Sin}[t]
  \]
  - \(V_b[t]\)：基底樹突膜電位  
  - \(V_a[t]\)：頂端樹突膜電位  
  - \(\sigma\)：sigmoid 函數（門控）  
  - 當 Soma 膜電位 \(U[t]\) 超過閾值時發放并重置。  

- **長短期記憶**：由於不同樹突之間的非線性互動與時間常數，可保留「歷史脈衝資訊」，提升序列任務的表現。  
- **可學習參數**：基底/頂端突觸權重 \(W_b, W_h^b, W_a, W_h^a\) 以及膜時間常數 \( \tau, \tau_a, \tau_b \)。  

---

##  四、程式執行與結果展示（Colab 測試）

###  環境與安裝

```bash
git clone https://github.com/BrainCog-X/Brain-Cog.git
cd Brain-Cog
pip install -r requirements.txt
pip install -e .
````

###  快速執行 CIFAR-10 模型（2 epochs 示範）

```bash
%cd /content/brain-cog/examples/Perception_and_Learning/img_cls/bp

python main.py \
  --model cifar_convnet \
  --dataset cifar10 \
  --node-type LIFNode \
  --step 8 \
  --device 0 \
  --epochs 2
```

#### 🖥 執行輸出示例

```
/content/brain-cog/examples/Perception_and_Learning/img_cls/bp
2025-06-03 09:45:14.743372: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1748943914.763165   15252 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1748943914.769445   15252 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-06-03 09:45:14.789885: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
...
[MODEL ARCH]
CifarConvNet(
  (encoder): Encoder()
  (vote): Identity()
  (feature): Sequential(
    (0): BaseConvModule(... conv → LIFNode → Identity)
    (1): BaseConvModule(... conv → LIFNode → Identity)
    (2): AvgPool2d(...)
    ...
    (11): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (fc): Sequential(
    (0): BaseLinearModule(Linear(1024, 10) → Identity)
  )
  (flatten): Flatten(start_dim=1)
)
learning rate is 0.000625
Model cifar_convnet created, param count: 19489562
[OPTIMIZER]
AdamW (...)
Scheduled epochs: 12
Train: 0 [   0/390 (  0%)]  Loss: 2.3183  Acc@1: 10.1562  Time: 3.18s  LR: 1.000e-06
...
Eval : 0  Loss: 1.9703  Acc@1: 35.94  Acc@5: 85.16
Current checkpoints:
 ('.../checkpoint-0.pth.tar', 30.16)

Train: 1 [   0/390 (  0%)]  Loss: 2.1501  Acc@1: 23.44  Time: 2.50s  LR: 1.258e-04
...
Eval : 1  Loss: 1.4132  Acc@1: 48.44  Acc@5: 92.97
Current checkpoints:
 ('.../checkpoint-1.pth.tar', 50.02)

*** Best metric: 50.02 (epoch 1) ***
```

> *   第一行顯示模型結構與參數數量
>     
> *   下一行顯示 optimizer 與學習率
>     
> *   之後依序印出每個 epoch 的 Train/Eval 結果
>     
> *   `checkpoint-0.pth.tar`（Epoch 0 最佳模型）Acc@1 約 30.16%
>     
> *   `checkpoint-1.pth.tar`（Epoch 1 最佳模型）Acc@1 約 50.02%
>     

###  訓練結果（2 epochs 範例）

| Epoch | Eval Loss | Eval Acc@1 | Eval Acc@5 |
| --- | --- | --- | --- |
| 0 | 1.9703 | 35.94 % | 85.16 % |
| 1 | 1.4132 | 48.44 % | 92.97 % |

> 上述數值僅為示例，實際會因硬體環境、隨機種子、batch size 等因素而異。

* * *

 五、結論與心得
----------

*   **Brain-Cog** 框架將 SNN 應用於多個領域（影像、語音、強化學習），展現了脈衝神經元在序列處理上的潛力。
    
*   **CifarConvNet** 示範了如何在 SNN 中設計多層卷積網絡，並成功完成 CIFAR-10 分類任務。
    
*   **MCN 多隔室神經元** 的核心思路在於模仿生物樹突非線性訊號整合，提供長短期記憶能力。
    
*   在 Colab 環境下，透過 `--epochs 2` 進行快速測試，即可取得可用結果；欲完整訓練建議將 `--epochs` 提高至 30 以上，並使用更強效的 GPU。
    

* * *

📎 參考資源
-------

*   [📄 論文 PDF (arXiv)](https://arxiv.org/pdf/2303.17469)
    
*   [📂 GitHub 原始碼](https://github.com/BrainCog-X/Brain-Cog)
    
*   [🌐 官方網站與文件](http://www.brain-cog.network/)
    

```

你可以將上述內容儲存為 `README.md` 並放到你的專案根目錄。 ​:contentReference[oaicite:0]{index=0}​
```
