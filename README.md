# Distilling Knowledge via Intermediate Classifier Heads (DIH)
<section>
  <h2>Table of Contents:</h2>
  <ul>
    <li><a href="#introduction"><b>Introduction</b></a></li>
      <ul>
        <li> <a href="#dih_pipeline"><b>DIH training pipeline</b></a></li>
      </ul>
    </li>
  <li><a href="#datasets"><b>Datasets</b></a></li>
  <li><a href="#baselines"><b>Baselines</b></a></li>
  <li><a href="#run"><b>Running The Experiments</b></a></li>
      <ul>
        <li><a href="#files"><b>Files in this repository</b></a></li>
        <li><a href="#hypers_section"><b>Hyper-parameters to set</b></a></li>
      </ul>
  <li><a href="#examples"><b>Example</b></a></li>
  <li><a href="#refs"><b>Reference</b></a></li>
    </li>
  </ul>
  <h2 id="introduction">Introduction</h2>
<p text-align: justify>
  <b>Distilling Knowledge via Intermediate Classifier Heads (DIH)</b> is a knowledge distillation framework that specifically tries to mitigate the negative impact of <i>the capacity gap</i>, i.e., the difference in model complexity between the teacher and the student model on knowledge distillation. This approach improves the canonical knowledge distillation (KD) with the help of the teacher's <i>intermediate representations</i> (the outputs of some of the hidden layers).
  <br>
  <h3 id="dih_pipeline">DIH training pipeline</h3>
  <ol>
  <li>First <b>k</b> classifier heads have to be mounted to various intermediate layers of the teacher (see <a href="#arch_table"">Table 1</a> for the structure of models, i.e., the location and also the value of <b>k</b> in this repository).</li>
  <li>The added intermediate classifier heads pass a cheap and efficient fine-tuning (while the main teacher is frozen). The fine-tuning step is quite cheaper and more efficient than training a whole model (i.e., a fraction of teacher model , and the added intermediate classifier head module) from scratch. This is due to the frozen state of the backbone of the model, i.e., inly the added intermediate head needs to be trained.</li>
  <li>The cohort of classifiers (all the mounted ones + the final main classifier) co-teach the student simultaneously with knowledge distillation.</li>
  </ol>
Our experiments on various teacher-student pairs of models and datasets have demonstrated that the proposed approach outperforms the canonical knowledge distillation approach and its extensions, which are intended to address the capacity gap problem.
  </section>
  <br>
  <section>
  <br>
    <div align="center"><img src="DIH.png" alt="Distilling Knowledge via Intermediate Classifier Heads (DIH)" width="600" height="500" align="center"></div>
  </section>
  <section>
  <h2>Requirements</h2>
  <ul>
    <li><b>torch 1.7.1</b> the project is built in PyTorch.</li>
    <li> <b>torchvision 0.8.2</b> used for datasets, and data preprocessing.</li>
  <li><b>tqdm 4.48.2</b> for better visualization of training process.</li>
  <li><b>torchsummary</b> for invesitating the model's architecture.</li>
   <li><b>numpy 1.19.4</b> used in preprocessing the dataset, adn showing examples.</li>
    <li><b>argparse</b> passsing the input variables for easy reproducibility.</li>
    <li><b>os</b> reading and writing the trained model's weights.</li>
 </ul>
  <code>pip3 install -r requirements.txt</code>
</section>

<section>
  <h2 id="datasets">Datasets</h2>
  <a href="https://www.cs.toronto.edu/~kriz/cifar.html"><b>CIFAR-10</b></a>, and <a href="https://www.cs.toronto.edu/~kriz/cifar.html"><b>CIFAR-100</b></a>  contain 32x32 pixel RGB images for 10 and 100 classes respectively. The datasets composed of 50,000 training and 10,000 testing images. All training and testing datasets are balanced (i.e., the number of images per class is the same within the dataset). For these two datasets, the images are augmented by the combination of horizontal flips, 4 pixels padding, and random crops. We also normalized the images by their mean and standard deviation. 
 </section>
 
 <section>
 <h2 id="baselines">Baselines</h2>
  <ol>
    <li><a href="https://arxiv.org/pdf/1503.02531.pdf">Canonical Knowledge Distillation (KD)</a></li> As one of the benchmarks, we use conventional knowledge distillation (in the context and the experiments, we have refered to canonical knowledge distillation as KD). We used the same temperature (τ=5), and same alpha weight(α=0.1) as DIH.
    <li><a href="https://arxiv.org/pdf/1412.6550.pdf">FitNets</a></li> FitNets as a knowledge distillation framework, first transfers the knowledge of a fraction of a trained teacher model up to a selected layer (known as hint layer) to a fraction of a student model up to a selected intermediate layer (called guided layer). This step optimizes the selected fraction of the student by using the L<sub>2</sub> loss objective. The second step of FitNets, is the canonical knowledge distillation (KD) to transfer the knowledge from the complete teacher to the complete student. For the first step, we trained the selected fraction of the student for 40 epochs using L<sub>2</sub> loss function. In the second step, we used the same setting for KD and we trained the complete student model for 200 epochs.
    <li><a href="https://arxiv.org/pdf/1902.03393.pdf">Knowledge Distillation with Teacher Assistants (TAKD)</a></li> We limited the number of teacher assistants to 1 for each experiment. The setting for training the teacher assistant and the final student is identical (the same setting for KD).
    <li><a href="https://arxiv.org/pdf/1706.00384.pdf">Deep Mutual Learning (DML)</a></li> Besides the previous benchmarks which are categorized as offline KD (i.e., the student is supervised by a fully trained teacher model), we also used deep mutual learning to evaluate our model with one of its online distillation counterparts (a scenario where there is no trained teacher, i.e., the teacher and the student co-teach each other). We set the same setting as KD, except we set the temperature (τ=1).
  </ol>
 </section>
 
 <section>
  <h2 id="run">Running The Experiments</h2>
  <li>First, the selected teacher model should be trained with regular cross-entropy with the hyper-parameters mentioned in <a href="#hypers_table">Table 2</a>.</li>
  <li>In the reported experiments, we have used three different seed values (3,10,21). All the experiments in the paper except those reported in Table 3 of the paper, are the average of three different runs.</li>
  <li>For each selected teacher, a number of mounted intermediate classifier heads need to be fine-tuned. The number of added intermediate heads for each model is available in the following table. In this repository, we have mounted each an intermediate classifier head after every group of residual and/or bottleneck blocks in ResNet family  models, and after each max pooling layer for VGG-11 model (<strong>Note:</strong> the VGG model has been equipped with batch normalization).
    <br>
    <div id="arch_table"><div>
      <br>
  <table style="width:400px" align="center">
    <caption text-align:center>Table 1. The number of mounted intermediate classifier heads to the models used in this repository.</caption>
  <tr>
  <th align="center">Teacher Model</th>
  <th align="center"># Intermediate heads (k)</th> 
   </tr>
    <tr>
    <td align="center">ResNet-34</td>
    <td align="center">4</td>
    </tr>
     <tr>
    <td align="center">ResNet-18</td>
    <td align="center">4</td>
    </tr>
     <tr>
    <td align="center">VGG-11</td>
    <td align="center">4</td>
    </tr>
     <tr>
    <td align="center">ResNet-110</td>
    <td align="center">3</td>
    </tr>
     <tr>
    <td align="center">ResNet-20</td>
    <td align="center">3</td>
    </tr>
     <tr>
    <td align="center">ResNet-14</td>
    <td align="center">3</td>
    </tr>
     <tr>
    <td align="center">ResNet-8</td>
    <td align="center">3</td>
    </tr>
</table>
<br>
  </li>
      
  <section>
  
  <h3 id="files">Files in this repository</h3>
  
  <ul>
  <li><code><b>dataload.py</b></code> loads the data loader for training, validation, and testing for both datasets (CIFAR10-CIFAR100).</li>
  <li><code><b>models_repo</b></code> contains model classes(two categories of ResNets, VGG, and also the intermediate classifier module).</li>
  <li><code><b>KD_Loss.py</b></code>  canonical knowledge distillation loss function.</li>
  <li><code><b>dih_utlis.py</b></code> includes the function for loading the trained intermediate heads.</li>
  <li><code><b>train_dih.py</b></code> contains the function for distillation via intermediate heads <b>(DIH)</b>.</li>
  <li><code><b>train_funcs.py</b></code> regular cross-entropy training, and intermediate header's fine_tuning functions.</li>
  <li><code><b>test.py</b></code> testing console for running the functions above.</li>
</ul
  <br>
  
 <h3 id="hypers_section">Hyper-parameters to set</h3>
  <br>
  <table style="width:400px" id="hypers_table" align="center">
  <caption>Table 2. list of all hyper-parameters, their argparse tags, and their assigned default values.</caption>
  <tr>
  <th align="center">Hyper-parameter</th>
  <th align="center">args tag</th> 
  <th align="center">Default value</th> 
   </tr>
    <tr>
    <td align="center">student model</td>
    <td align="center">student</td>
    <td align="center">res8</td>
    </tr>
     <tr>
    <td align="center">teacher model</td>
    <td align="center">teacher</td>
    <td align="center">res110</td>
    </tr>
    <tr>
    <td align="center">learning rate</td>
    <td align="center">lr</td>
    <td align="center">0.1</td>
    </tr>
     <tr>
    <td align="center">weight decay</td>
    <td align="center">wd</td>
    <td align="center">5e-4</td>
    </tr>
     <tr>
    <td align="center">epochs</td>
    <td align="center">epochs</td>
    <td align="center">200</td>
    </tr>
     <tr>
    <td align="center">dataset</td>
    <td align="center">dataset</td>
    <td align="center">cifar100</td>
    </tr>
     <tr>
    <td align="center">schedule</td>
    <td align="center">schedule</td>
    <td align="center">[60,120,180]</td>
    </tr>
    <tr>
    <td align="center">γ</td>
    <td align="center">schedule_gamma</td>
    <td align="center">0.1</td>
    </tr>
     <tr>
    <td align="center">temperature τ (KD)</td>
    <td align="center">kd_temperature</td>
    <td align="center">5</td>
    </tr>
     <tr>
    <td align="center">α (KD)</td>
    <td align="center">kd_alpha</td>
    <td align="center">0.1</td>
    </tr>
      <tr>
    <td align="center">batch size</td>
    <td align="center">batch_size</td>
    <td align="center">64</td>
    </tr>
     <tr>
    <td align="center">training type</td>
    <td align="center">training_type</td>
    <td align="center">dih</td>
  </tr>
</table>
<br> 
  </section>
</section>

  
  <section>
  <h2 id="examples">Example</h2>
  </li>
      <div id="ce_template">
       For training a model with <b>regular cross-entropy</b> the following template should be run:
    <br>
     <code>python3 final_test.py --training_type ce --teacher res110 --path_to_save /home/teacher.pth --batch_size 64 --dataset cifar100 --epochs 200 --gpu_id cuda:0 --lr 0.1 --schedule [60,120,180] --wd 0.0005 --seed 3</code>
  </div>
   <br>
      <div id="fine_tune_template">
        By having a trained teacher, we need to <b>fine_tune</b> all of its intermediate classifier heads by running the following command:
    <br>
     <code>python3 final_test.py  --training_type fine_tune --teacher res110 --path_to_save /home/headers --batch_size 64 --dataset cifar100  --epochs 200 --gpu_id cuda:0  --lr 0.1 --schedule [60,120,180] --wd 0.0005 --seed 3</code>
      </div>
  <br>
        <div id="fitents">
          For evaluation, we used <b>FitNets</b>. To train a student with this approach, this template should be runned:
    <br>
     <code>python3 final_test.py --student res8 --training_type fitnets --teacher res110 --saved_path /home/teacher.pth  --path_to_save /home/stage_1.pth --epochs_fitnets_1 40  nesterov_fitnets_1 True --momentum_fitnets_1 0.9 --lr_fitnets_1 0.1 --wd_fitnets_1 0.0005 --batch_size 64  --dataset cifar100 --epochs 200 --gpu_id cuda:0  --lr 0.1 --schedule [60,120,180] --wd 0.0005 --seed 3</code>
      </div>
  <br>
         <div id="dml">
       As another baseline, one can train a cohort of two models (student and the teacher) as a cohort via <b>deep mutual learning (DML)</b>, by the following template:
    <br>
     <code>python3 final_test.py --student res8 --training_type dml --teacher res110 --path_to_save /home/dml --batch_size 64 --dataset cifar100 --epochs 200 --gpu_id cuda:0  --lr 0.1 --schedule [60,120,160] --wd 0.0005 --alpha 0.1 --temperature 1 --seed 3</code>
      </div>
  <br>
            <div id="kd">
              The canonical <b>knowledge distillation (KD)</b> is available through the following command:  
    <br>
     <code>python3 final_test.py --student res8 --training_type kd --teacher res110 --saved_path /home/teacher.pth --path_to_save /home/res8_kd.pth --batch_size 64 --dataset cifar100 --epochs 200 --gpu_id cuda:0  --lr 0.1 --schedule [60,120,180] --wd 0.0005 --alpha 0.1 --seed 3 --temperature 5 </code>
      </div>
  <br> 
  The student Resnet-8 can be trained via <b>DIH</b> through the following command:
  <code>python3 final_test.py --student res8 --teacher res110 --saved_path /home/teacher.pth --saved_intermediates_directory /home/saved_headers/ --alpha 0.1  --temperature 5 --batch_size 64  --dataset cifar100  --epochs 200 --gpu_id cuda:0  --lr 0.1 --schedule [60, 120, 180] --wd 0.0005 --seed 3 --path_to_save /home/dih_model.pth
</code>
</section>

<section>
  
  <h2 id="refs">Reference</h2>
  <p>
  arxiv link: <a href="http://arxiv.org/abs/2103.00497">http://arxiv.org/abs/2103.00497</a><br>
  If you found this library useful in your research, please consider citing:
  <br>
  <div>
</div>
<embed src="cite.bib" height="200" width="300"></embed>  

  
 </p>
  </section>
