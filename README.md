# Distilling Knowledge via Intermediate Classifier Heads (DIH)
<section>
<p text-align: justify>
  <b>Distilling Knowledge via Intermediate Classifier Heads (DIH)</b> is a knowledge distillation framework which specifically tries to mitigate the negative impact of <b>capacity gap</b> between the teacher and the student model on knowledge distillation. This approach improves the canonical knowledge distillation (KD) with the help of teacher's <b>intermediate representations</b>.
  <br>
  DIH pipeline:
  <ol>
  <li>First <b>k</b> classifier heads have to be mounted at various intermediate layers of the tecaher.</li>
  <li>The added intermediate classifier heads pass a <b>cheap</b> fine-tuning ( while the main teacher is forzen).</li>
  <li>The cohort of classifiers (all the mounted ones + the final main classifier) co-teach the student simultaneously with knowledge distillation.</li>
  </ol>
Our experiments on various teacher-student pairs of models and datasets have demonstrated that the proposed approach outperforms the canonical knowledge distillation approach and its extensions, which are intended to address the capacity gap problem.
  <br>
    <img src="DIH.png" alt="Distilling Knowledge via Intermediate Classifier Heads (DIH)"width: 60% height: 60% justify-content: center>
  </section>
  <section>
  <h2>Requirements</h2>
  <ul>
  <li>torch==1.7.1</li>
  <li>torchvision==0.8.2</li>
  <li>tqdm==4.48.2</li>
  <li>torchsummary</li>
   <li>numpy==1.19.4</li>
 </ul>
  <code>pip3 install -r requirements.txt</code>
</section>

<section>
  <h2>Datasets</h2>
   <ul>
  <li><a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10</a></li>
  <li><a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-100</a></li>
 </ul>
 </section>
 
 <section>
  <h2>Instructions</h2>
  <li>First the a selected teacher model should be trained with regular cross-entropy with the hyper-parameters mentioned above.</li>
  <li>For each selected teacher, a number of mounted <b>intermediate classifier heads</b> need to be fine-tuned. The number of added intermediate heads for each model are available in the following table.
    
    <style>
table {
  width:100%;
}
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding: 15px;
  text-align: left;
}
#t01 tr:nth-child(even) {
  background-color: #eee;
}
#t01 tr:nth-child(odd) {
 background-color: #fff;
}
#t01 th {
  background-color: black;
  color: white;
}
</style>
  
  
  
  </li>
 </section>
  
  <section>
  <h2>Example</h2>
  <p>Student=ResNet8, Teacher=ResNet110, CIFAR-100  </p>
  <code>python3 test_dih.py --alpha 0.1  --batch_size 64  --dataset cifar100  --epochs 200 --gpu_id 0  --lr 0.1 --schedule [60, 120, 180] --temperature 5 --wd 0 .0005
</code>
  

  
</section>
