<div style="padding-top: 20px;"> </div>
<h1><a id="giga-ai" class="anchor" aria-hidden="true" href="#giga-ai"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>
AI-enabled School Mapping Model Results  </h1> 

<details open="open">
	<summary style="padding-bottom: 10px;"><h2>Table of Contents</h2></summary>
  <ol>
    <li><a href="#benin">Benin (BEN)</a></li>
    <li><a href="#botswana">Botswana (BWA)</a></li>
    <li><a href="#ghana">Ghana (GHA) </a></li>
    <li><a href="#kenya">Kenya (KEN) </a></li>
    <li><a href="#malawi">Malawi (MWI) </a></li>
    <li><a href="#mongolia">Mongolia (MNG) </a></li>
    <li><a href="#mongolia">Mozambique (MOZ) </a></li>
    <li><a href="#namibia">Namibia (NAM) </a></li>
    <li><a href="#rwanda">Rwanda (RWA) </a></li>
    <li><a href="#senegal">Senegal (SEN) </a></li>
    <li><a href="#south-sudan">South Sudan (SSD) </a></li>
    <li><a href="#tajikistan">Tajikistan (TJK) </a></li>
    <li><a href="#zimbabwe">Zimbabwe (ZWE) </a></li>
</details>

<h2><a id="model-results" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Model Results</h2>

<h3><a id="benin" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Benin (BEN)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| swin_v2_s | 0.983 |
| vit_h_14 | 0.978 | 
| convnext_base | 0.977 | 
| vsc-ensemble | 0.998 |

|Best CAM Method | Probability Threshold | F2 Score
|---|:---:|:---:|
| GradCamElementWise | 0.366 | 0.982 |  |

<h3><a id="botswana" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Botswana (BWA)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_l_16 | 0.989 | 
| convnext_large | 0.985 | 
| swin_v2_t | 0.984 | 
| vsc_ensemble | 0.997 |

|Best CAM Method | Prob Threshold | F2 Score
|---|:---:|:---:|
| GradCamElementWise| 0.352 | 0.960 | 

<h3><a id="ghana" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Ghana (GHA)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC 
|---|:---:| 
| swin_v2_s | 0.931 | 
| vit_h_14 | 0.930 | 
| convnext_small | 0.928 | 
| vsc-ensemble | 0.991 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCamElementWise | 0.386 | 0.968 | 


<h3><a id="kenya" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Kenya (KEN)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| convnext_small | 0.916 |
| swin_v2_b | 0.910 | 
| vit_b_16 | 0.906 | 
| vsc_ensemble | 0.966 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCamElementWise | 0.395 | 0.966 | 


<h3><a id="malawi" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Malawi (MWI)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| convnext_base | 0.969 |
| vit_h_14 | 0.967 | 
| swin_v2_s | 0.962 | 
| vsc_ensemble | 0.983 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCamElementWise | 0.335 | 0.953 | 

<h3><a id="mongolia" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Mongolia (MNG)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv2</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv2">https://wandb.ai/issatingzon/GIGAv2</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_b_16 | 0.950 |
| convnext_base| 0.944 | 
| swin_v2_b | 0.935 | 
| vsc_ensemble | 0.991 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| Hirescam | 0.570 | 0.938 | 

<h3><a id="mozambique" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Mozambique (MOZ)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv2</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv2">https://wandb.ai/issatingzon/GIGAv2</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| convnext_small | 0.969 |
| swin_v2_b| 0.968| 
| vit_h_14| 0.965| 
| vsc_ensemble | 0.994 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCamElementWise | 0.377 | 0.974 | 

<h3><a id="namibia" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Namibia (NAM)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_h_14 | 0.955 |
| convnext_large | 0.954| 
| swin_v2_s| 0.949| 
| vsc_ensemble | 0.980| 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCamElementWise | 0.315 | 0.914| 

<h3><a id="rwanda" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Rwanda (RWA)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_h_14| 0.983 |
| swin_v2_t| 0.982| 
| convnext_base| 0.978| 
| vsc_ensemble | 0.998| 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCam| 0.344 | 0.978| 

<h3><a id="senegal" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Senegal (SEN)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_h_14 | 0.980 |
| convnext_large | 0.978| 
| swin_v2_t| 0.967| 
| vsc_ensemble | 0.993| 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCam| 0.355| 0.985 | 


<h3><a id="south-sudan" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>South Sudan (SSD)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_h_14 | 0.971 |
| convnext_base| 0.964| 
| swin_v2_t| 0.964 | 
| vsc_ensemble | 0.995 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCam | 0.378 | 0.962| 

<h3><a id="tajikistan" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Tajikistan (TJK)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv3</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv3">https://wandb.ai/issatingzon/GIGAv3</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| convnext_large | 0.962 |
| vit_h_14 | 0.955 | 
| swin_v2_s| 0.944| 
| vsc_ensemble | 0.986 | 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCamElementWise| 0.342| 0.967 | 


<h3><a id="zimbabwe" class="anchor" aria-hidden="true" href="#overview"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"></path></svg></a>Zimbabwe (ZWE)</h3>
<ul>
<li> <b>Project:</b> <code>GIGAv1</code> </li>
<li> <b>WandB:</b> <a href="https://wandb.ai/issatingzon/GIGAv1">https://wandb.ai/issatingzon/GIGAv1</a></li>
</ul>

| Best Models | Test AUPRC | 
|---|:---:| 
| vit_h_14 | 0.971 |
| convnext_base| 0.967| 
| swin_v2_b| 0.961| 
| vsc_ensemble | 0.996| 

|Best CAM Method | Prob Threshold | F2 Score 
|---|:---:|:---:|
| GradCam++| 0.327| 0.977 | 

