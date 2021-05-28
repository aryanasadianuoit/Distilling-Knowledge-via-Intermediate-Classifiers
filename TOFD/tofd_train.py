from tofd_utils import *
from resnet_cifar_tofd import *
from models.OOG_resnet import ResNet34,ResNet18
from models.MobileNetV2_CIFAR import MobileNetV2
from dataloader import get_train_valid_loader_cifars,get_test_loader_cifar
from benchmarks.tofd.tofd_resnets import resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Task-Oriented Feature Distillation. ')
parser.add_argument('--model', default="res20", help="choose the student model", type=str)
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar10/cifar100")
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.03, type=float)
parser.add_argument('--l2', default=7e-3, type=float)
parser.add_argument('--teacher', default="vgg11", type=str)
parser.add_argument('--t', default=3.0, type=float, help="temperature for logit distillation ")
parser.add_argument('--seed', default=67, type=int, help="Seed value for reproducibility")
args = parser.parse_args()
print(args)

BATCH_SIZE = 128
LR = 0.1
"""
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset, testset = None, None
if args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4)
"""

if args.dataset == "cifar10":

    data_loader_dict, dataset_sizes = get_train_valid_loader_cifars(batch_size=BATCH_SIZE,
                                                                    cifar10_100=args.dataset)

    train_loader = data_loader_dict["train"]

    test_loader = get_test_loader_cifar(batch_size=BATCH_SIZE,
                                        dataset=args.dataset)


    NUM_ClASSES=10


elif args.dataset == "cifar100":
    data_loader_dict, dataset_sizes = get_train_valid_loader_cifars(batch_size=BATCH_SIZE,
                                                                    cifar10_100=args.dataset)
    train_loader = data_loader_dict["train"]

    test_loader = get_test_loader_cifar(batch_size=BATCH_SIZE,
                                        dataset=args.dataset)
    NUM_ClASSES = 100

elif args.dataset == "tiny":
    #TODO
    NUM_ClASSES = 200

#   get the student model
if args.model == "res18":
    net = ResNet18(seed=args.seed,num_classes=NUM_ClASSES)
if args.model == "res34":
    net = ResNet34(seed=args.seed,num_classes=NUM_ClASSES)
if args.model == "res110":
    net = resnet110_cifar(seed=args.seed,num_classes=NUM_ClASSES)
if args.model == "res32":
    net = resnet32_cifar(seed=args.seed,num_classes=NUM_ClASSES)
if args.model == "res20":
    net = resnet20_cifar(seed=args.seed,num_classes=NUM_ClASSES)
if args.model == "res8":
    net = resnet8_cifar(seed=args.seed, num_classes=NUM_ClASSES)
if args.model == "mobilenet":
    net = MobileNetV2(seed=args.seed, num_classes=NUM_ClASSES)
if args.model == 'wres28_2':
    from wres_net_tofd import get_Wide_ResNet_28_2_tofd
    net = get_Wide_ResNet_28_2_tofd(seed=args.seed, num_classes=NUM_ClASSES)

#   get the teacher model
if args.teacher == 'resnet34':
    teacher = resnet34(num_classes=NUM_ClASSES)
elif args.teacher == 'wres28_2':
    from wres_net_tofd import get_Wide_ResNet_28_2_tofd
    teacher = get_Wide_ResNet_28_2_tofd(seed=args.seed,num_classes=NUM_ClASSES)

elif args.teacher == 'vgg11':
    from VGG_TOFD import VGG_Intermediate_Branches_TOFD
    teacher = VGG_Intermediate_Branches_TOFD("VGG11",seed=args.seed,num_classes=NUM_ClASSES)
elif args.teacher == 'res110':
    teacher = resnet110_cifar(seed=args.seed,num_classes=NUM_ClASSES)
elif args.teacher == 'resnet20':
    teacher = resnet20_cifar(seed=args.seed,num_classes=NUM_ClASSES)


#teacher.load_state_dict(torch.load("./teacher/" + args.teacher + ".pth"))
#teacher.load_state_dict(torch.load("/home/aasadian/virtualvenvs/gputestvenv/fitnes_from_scratch/codistillation/bests/ce/res110_cifar100.pth"))
#saved_teacher_state_dict = ()
#temp_dict = {}

#teacher_path = "/home/aasadian/tofd/teacher/teacher_res110.pth"
teacher_path = "/home/aasadian/tofd/teacher/teacher_vgg11_tofd_seed_50.pth"

full_modules_state_dict = {}
saved_state_dict = torch.load(teacher_path)
testing_state_dict = {}
for (key, value), (key_saved, value_saved) in zip(teacher.state_dict().items(), saved_state_dict.items()):
    testing_state_dict[key] = value_saved
    full_modules_state_dict["core." + key] = value_saved
teacher.load_state_dict(testing_state_dict)
#teacher_core.eval()


teacher.cuda()
net.to(device)
orthogonal_penalty = args.beta
init = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    print("Start Training")
    for epoch in range(200):
    #for epoch in range(250):
        if epoch in [60, 120, 180]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, student_feature = net(inputs)
            #student_feature = out1,out2,out3

            #   get teacher results
            with torch.no_grad():
                teacher_logits, teacher_feature = teacher(inputs)

            #   init the feature resizing layer depending on the feature size of students and teachers
            #   a fully connected layer is used as feature resizing layer here
            if not init:
                teacher_feature_size = teacher_feature[0].size(1)
                student_feature_size = student_feature[0].size(1)
                num_auxiliary_classifier = len(teacher_logits)
                link = []
                for j in range(num_auxiliary_classifier):
                    link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
                net.link = nn.ModuleList(link)
                net.cuda()
                #   we redefine optimizer here so it can optimize the net.link layers.
                optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)
                init = True


            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   Distillation Loss + Task Loss
            for index in range(len(student_feature)):

                student_feature[index] = net.link[index](student_feature[index])
                #   task-oriented feature distillation loss
                loss += torch.dist(student_feature[index], teacher_feature[index], p=2) * args.alpha
                #   task loss (cross entropy loss for the classification task)
                loss += criterion(outputs[index], labels)
                #   logit distillation loss, CrossEntropy implemented in utils.py.
                loss += CrossEntropy(outputs[index], teacher_logits[index], 1 + (args.t / 250) * float(1 + epoch))

            # Orthogonal Loss
            for index in range(len(student_feature)):
                weight = list(net.link[index].parameters())[0]
                weight_trans = weight.permute(1, 0)
                ones = torch.eye(weight.size(0)).cuda()
                ones2 = torch.eye(weight.size(1)).cuda()
                loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * args.beta
                loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * args.beta

            sum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(labels.size(0))
            _, predicted = torch.max(outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())

            if i % 20 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                     100 * correct / total))

    print("Waiting Test!")
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for data in test_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, feature = net(images)
            _, predicted = torch.max(outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))

        print('Test Set AccuracyAcc:  %.4f%% ' % (100 * correct / total))
        if correct / total > best_acc:
            best_acc = correct / total
            print("Best Accuracy Updated: ", best_acc * 100)
            torch.save(net.state_dict(), "/home/aasadian/tofd/" + args.model+"_teacher_"+ args.teacher + ".pth")
print("Training Finished, Best Accuracy is %.4f%%" % (best_acc * 100))