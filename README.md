# Distilling Knowledge via Intermediate Classifier Heads (DIH)
<p text-align: justify>
The crux of knowledge distillation—as a model
compression and transfer-learning approach—is
to train a resource-limited student model with
the guide of a pre-trained larger teacher model
to tighten the accuracy gap between the teacher
and the student. However, when there is a large
gap between the model complexities of teacher
  and student (also known as <b>capacity gap</b>), knowledge
distillation loses its strength in transferring
knowledge from the teacher to the student, thus
training a weaker student. To mitigate the impact
of the capacity gap on knowledge distillation,
we introduce <b>knowledge distillation via intermediate
  heads</b>. We first cheaply acquire a cohort
of pre-trained teachers with various model complexities,
by extending the intermediate layers
of the teacher (at various depth) with classifier
heads. These intermediate classifier heads can
all together be efficiently learned while freezing
the backbone of the pre-trained teacher. The cohort
of teachers (including the original teacher)
co-teach the student simultaneously. Our experiments
on various teacher-student pairs of models
and datasets have demonstrated that the proposed
approach outperforms the canonical knowledge
distillation approach and its extensions, which are
intended to address the capacity gap problem.
  </p>
  
  <section>
  <h2>Base Line</h2>
  <p>Student= ResNet8, Teacher = ResNet110  </p>
  <br>
    <code>python3 test_dig.py --alpha 0.1  --batch_size 64  --dataset cifar100  --epochs 200 --gpu_id 0  --lr 0.1 --schedule [60, 120, 180] --temperature 5 --wd 0 .0005
</code>
  

  
</section>
