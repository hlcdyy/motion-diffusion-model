from openTSNE import TSNE
import os 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from utils.parser_util import eval_motion_encoder_args 
from utils.model_util import create_motion_encoder, load_model_wo_clip
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
import data_loaders.humanml.utils.paramUtil as paramUtil
from tqdm import tqdm
from utils.fixseed import fixseed
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    args = eval_motion_encoder_args()
    fixseed(args.seed)
    max_frames = 196 if args.dataset in ['kit', 'humanml', "bandai-2_posrot", "bandai-1_posrot"] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)

    print('Loading dataset...')
    if args.dataset in ["bandai-1_posrot", "bandai-2_posrot"]:
        data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames, split='all')

    model = create_motion_encoder(args, data)

    motionenc_dict = torch.load(args.model_path, map_location='cpu')
    print("load motion enc model: {}".format(args.model_path))
    
    load_model_wo_clip(model, motionenc_dict)

    model.to(dist_util.dev())
    model.eval()

    # style_label = "elderly"
    content_fix = "wave both hands"
    content_labels = []
    all_hiddens = []
    for motion, cond in tqdm(data):
        motion = motion.to(dist_util.dev())
        captions = cond["y"]["text"]
        all_contents = []
        for caption in captions:
            content = caption.split(" ")[2:-1]
            content[0] = content[0][:-1]
            content = " ".join(content)
            all_contents.append(content)
        cond["y"] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in cond["y"].items()}
        with torch.no_grad():
            hiddens, _ = model(motion, **cond) # batch_size, hidden_size
        # for i, style in enumerate(cond["y"]["style"]):
        #     if style == style_label:
                # all_hiddens.append(hiddens[i])
                # content_labels.append(all_contents[i])
        for i, content in enumerate(all_contents):
            if content == content_fix:
                all_hiddens.append(hiddens[i])
                content_labels.append(cond["y"]["style"][i])
        
    all_hiddens = torch.stack(all_hiddens, 0).detach().cpu().numpy()
    distinct_content = list(set(content_labels))

    tsne = TSNE(
        perplexity=300,
        metric="euclidean",
        n_jobs=16,
        random_state=42,
        verbose=True,
    )
    embedding = tsne.fit(all_hiddens) # sample * 2
    
    groups = [[] for _ in range(len(distinct_content))]
    for j, content in enumerate(distinct_content):
        for i, label in enumerate(content_labels):
            if label == content:
                groups[j].append(embedding[i, :])

    for i in range(len(distinct_content)):
        groups[i] = np.stack(groups[i], 0)

    fig = plt.figure(1)
    cmap = plt.cm.get_cmap('inferno')
    color_map = cmap(np.linspace(0, 1, len(distinct_content)))
    for i in range(len(distinct_content)):
        plt.scatter(groups[i][:, 0], groups[i][:, 1], color=color_map[i, :], s=3, label=distinct_content[i])

    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))    # plt.legend()
    plt.gca().add_artist(legend)
    # plt.title('Style {}'.format(style_label))
    plt.title('Content {}'.format(content_fix))
    directory = os.path.split(args.model_path)
    # path = os.path.join(directory[0], "tsne_"+style_label+".png")
    path = os.path.join(directory[0], "tsne_"+content_fix+".png")
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("visual")
    

if __name__ == "__main__":
    main()
        
        

    

