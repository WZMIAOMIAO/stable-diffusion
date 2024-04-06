# Stable Diffusion
*感谢与[Stability AI](https://stability.ai/)以及[Runway](https://runwayml.com/)的合作，再加上我们先前的一些工作使我们能够做出Stable Diffusion：*

[**High-Resolution Image Synthesis with Latent Diffusion Models**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Dominik Lorenz](https://github.com/qp-qp)\,
[Patrick Esser](https://github.com/pesser),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
_[CVPR '22 Oral](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
[GitHub](https://github.com/CompVis/latent-diffusion) | [arXiv](https://arxiv.org/abs/2112.10752) | [Project page](https://ommer-lab.com/research/latent-diffusion-models/)_

![txt2img-stable2](assets/stable-samples/txt2img/merged-0006.png)
[Stable Diffusion](#stable-diffusion-v1)是一个隐式的文生图扩散模型。感谢[Stability AI](https://stability.ai/)提供了充足的算力支持，以及[LAION](https://laion.ai/)提供的数据支撑，使得我们能够在[LAION-5B](https://laion.ai/blog/laion-5b/)数据集上采用`512x512`分辨率训练一个隐式扩散模型。和谷歌的[Imagen](https://arxiv.org/abs/2205.11487)类似，该模型使用了一个冻结权重的CLIP ViT-L/14文本编码器，利用该编码器得到控制模型的文本提示。该模型包含了一个860M UNet以及一个123M text encoder，相对而言比较轻量使用一个拥有10G显存的GPU就能运行。

  
## Requirements
可通过如下指令创建一个名为`ldm`的[conda](https://conda.io/)环境：
```
conda env create -f environment.yaml
conda activate ldm
```

创建好上述环境后，你也可以通过如下指令更新[latent diffusion](https://github.com/CompVis/latent-diffusion)环境：

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
``` 


## Stable Diffusion v1
Stable Diffusion v1采用的是扩散模型架构，其主要由三部分组成: (1) 下采样倍率为8的自编码器 (2) 参数规模860M的UNet (3) CLIP ViT-L/14文本编码器。Stable Diffusion v1是先在`256x256`分辨率的图像上进行预训练，然后在`512x512`分辨率的图像上进行finetune。

*注意: Stable Diffusion v1是一个通用的文生图扩散模型，因此生成的结果会带上来自训练数据中的偏见和错误概念。有关训练过程，训练数据，模型用途等相关细节可参考[model card](Stable_Diffusion_v1_Model_Card.md).*

权重文件可通过[the CompVis organization at Hugging Face](https://huggingface.co/CompVis)下载，并遵守[The CreativeML OpenRAIL M license](LICENSE)条款。虽然该条款允许商业使用，**但我们不建议在没有额外安全机制以及考虑的情况下使用开源权重**。因为该模型权重存在[已知的限制和偏见](Stable_Diffusion_v1_Model_Card.md#limitations-and-bias)，并且关于通用文生图模型的安全、道德部署研究正在进行中。**模型权重是研究探索过程中的产物，理应如此看待**。

[The CreativeML OpenRAIL M license](LICENSE) is an [Open RAIL M license](https://www.licenses.ai/blog/2022/8/18/naming-convention-of-responsible-ai-licenses), adapted from the work that [BigScience](https://bigscience.huggingface.co/) and [the RAIL Initiative](https://www.licenses.ai/) are jointly carrying in the area of responsible AI licensing. See also [the article about the BLOOM Open RAIL license](https://bigscience.huggingface.co/blog/the-bigscience-rail-license) on which our license is based.

### Weights

当前，我们提供了如下权重：

- `sd-v1-1.ckpt`: 先在[laion2B-en](https://huggingface.co/datasets/laion/laion2B-en)数据集上采用`256x256`分辨率训练迭代了237k steps，接着在[laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution)数据集上采用`512x512`分辨率训练迭代了194k steps.
- `sd-v1-2.ckpt`: 基于`sd-v1-1.ckpt`权重在[laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/)数据集上采用`512x512`分辨率又训练迭代了515k steps（`laion-aesthetics v2 5+`是`laion2B-en`的子集，该子集美学评估分数大于`5.0`，过滤掉了原始分辨率大于等于`512x512`的图像，水印评估概率小于`0.5`，其中水印评估概率来自[LAION-5B](https://laion.ai/blog/laion-5b/) metadata，美学评估分数使用[LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor)计算得到）
- `sd-v1-3.ckpt`: 基于`sd-v1-2.ckpt`权重在"laion-aesthetics v2 5+"数据集上采用`512x512`分辨率接着训练迭代了195k step，并且训练过程中会丢弃10\%的文本条件信息以提升[classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).
- `sd-v1-4.ckpt`: 基于`sd-v1-2.ckpt`权重在"laion-aesthetics v2 5+"数据集上采用`512x512`分辨率接着训练迭代了225k step，并且训练过程中会丢弃10\%的文本条件信息以提升[classifier-free guidance sampling](https://arxiv.org/abs/2207.12598).

下图展示了不同的训练权重采用不同的classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,
5.0, 6.0, 7.0, 8.0)以及固定50步的PLMS sampling结果:
![sd evaluation results](assets/v1-variants-scores.jpg)



### Text-to-Image with Stable Diffusion
![txt2img-stable2](assets/stable-samples/txt2img/merged-0005.png)
![txt2img-stable2](assets/stable-samples/txt2img/merged-0007.png)

Stable Diffusion是由CLIP ViT-L/14文本编码器生成的text embeddings控制的隐式扩散模型。
我们提供了一个[可供参考的采样脚本](#reference-sampling-script), 但同时这也有一个已存在的[diffusers integration库](#diffusers-integration), 我们期待这个库对应的开源社区能够更加活跃。

#### Reference Sampling Script
我们提供了一个参考脚本，其中包括了

- 一个[安全检查模块](https://github.com/CompVis/stable-diffusion/pull/36),
  以避免生成一些不好的内容。
- 在输出图像中加入了[隐藏水印](https://github.com/ShieldMnt/invisible-watermark)
  可帮助查看者[鉴别该图像是否由机器生成](scripts/tests/test_watermark.py)。

在[获取到`stable-diffusion-v1-*-original`权重](#weights)后, 通过软连接的形式链接它。
```
mkdir -p models/ldm/stable-diffusion-v1/
ln -s <path/to/model.ckpt> models/ldm/stable-diffusion-v1/model.ckpt 
```
接着使用如下指令进行采样：
```
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms 
```

这里默认使用的是[Katherine Crowson's implementation](https://github.com/CompVis/latent-diffusion/pull/51)的[PLMS](https://arxiv.org/abs/2202.09778) 采样器，并且将guidance scale默认设置为`--scale 7.5`, 并且生成的图像分辨率是`512x512`（保持和训练时一致）且采样步数为50步。完整的参数可通过(`python scripts/txt2img.py --help`)查看。

```commandline
usage: txt2img.py [-h] [--prompt [PROMPT]] [--outdir [OUTDIR]] [--skip_grid] [--skip_save] [--ddim_steps DDIM_STEPS] [--plms] [--laion400m] [--fixed_code] [--ddim_eta DDIM_ETA]
                  [--n_iter N_ITER] [--H H] [--W W] [--C C] [--f F] [--n_samples N_SAMPLES] [--n_rows N_ROWS] [--scale SCALE] [--from-file FROM_FILE] [--config CONFIG] [--ckpt CKPT]
                  [--seed SEED] [--precision {full,autocast}]

optional arguments:
  -h, --help            展示帮助文档信息
  --prompt [PROMPT]     输入想要生成图像的描述语句
  --outdir [OUTDIR]     指定生成结果保存目录
  --skip_grid           跳过将所有生成图片保存成一张大图，当需要评估大批量生成图像时有用
  --skip_save           跳过单独保存每张生成图像（在测速时可跳过）
  --ddim_steps DDIM_STEPS 扩散模型的迭代次数
  --plms                使用plms sampling
  --laion400m           uses the LAION400M model
  --fixed_code          if enabled, uses the same starting code across samples
  --ddim_eta DDIM_ETA   ddim eta (eta=0.0 corresponds to deterministic sampling
  --n_iter N_ITER       sample this often
  --H H                 生成图像的高度
  --W W                 生成图像的宽度
  --C C                 隐空间的通道数
  --f F                 图像到隐空间的下采样倍率
  --n_samples N_SAMPLES
                        推理时采用的batch_size可以理解同时生成图像的数目, 如果显存少可设置小点
  --n_rows N_ROWS       rows in the grid (default: n_samples)
  --scale SCALE         unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
  --from-file FROM_FILE
                        if specified, load prompts from this file
  --config CONFIG       path to config which constructs model
  --ckpt CKPT           path to checkpoint of model
  --seed SEED           the seed (for reproducible sampling)
  --precision {full,autocast}
                        evaluate at this precision
```
注意：在`configs/stable-diffusion/v1-inference.yaml`配置中已将`use_ema`设置为`False`，这样你可以下载不带`EMA`的权重（文件大小会小很多），如果你不知道如何区分有没有带`EMA`的权重可以直接下载带有`full`关键字的权重例如`sd-v1-4-full-ema.ckpt`，这样在载入权重时会自动根据`use_ema`参数载入对应权重。

#### Diffusers Integration
另一种更简单的方式是直接使用[diffusers库](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers)，具体环境配置可参考[diffusers library](https://github.com/huggingface/diffusers/tree/main#new--stable-diffusion-is-now-fully-compatible-with-diffusers)：
```py
# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  
    
image.save("astronaut_rides_horse.png")
```


### Image Modification with Stable Diffusion
通过[SDEdit](https://arxiv.org/abs/2108.01073)提出的扩散去噪机制，该模型适用于多种任务，例如基于本文引导的图生图转换，图像超分等。与txt2img脚本类似，我们提供了一个利用Stable Diffusion对图像进行转换生成的脚本。

在下面示例中，利用一张手绘草图转换生成出一张充满细节的艺术品。
```
python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8
```
其中strength是一个0-1之间的参数，该参数用于控制添加到原图上噪声的比例。数值越接近1生成的多样性就越高但语义不一致的现象越明显。下面是一些示例。

**Input**

![sketch-in](assets/stable-samples/img2img/sketch-mountains-input.jpg)

**Outputs**

![out3](assets/stable-samples/img2img/mountains-3.png)
![out2](assets/stable-samples/img2img/mountains-2.png)

## Comments 

- 我们扩散模型代码主要是建立在[OpenAI's ADM codebase](https://github.com/openai/guided-diffusion)以及[https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)。开源万岁！

- transformer encoder的实现来自[lucidrains](https://github.com/lucidrains?tab=repositories)的[x-transformers](https://github.com/lucidrains/x-transformers)。

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


