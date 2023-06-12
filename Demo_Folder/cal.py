import torch
from transformers import AutoProcessor, AutoModel
import csv


import numpy as np
import torch
import os
from PIL import Image
import argparse
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval()
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def textual_alignment_score(frames, prompt, device):
    inputs = processor_clip(text=[prompt], images=frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model_clip(**inputs)
    logits_per_image = outputs.logits_per_image.detach().cpu().numpy()
    image_clip_score = logits_per_image.mean()

    return image_clip_score


def calc_probs(prompt, images,device):

    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        # probs = torch.softmax(scores, dim=-1)
    
    # return probs.cpu().tolist()
    scores_list = scores.cpu().tolist()
    return sum(scores_list)/len(scores_list)

def frame_consistency_score(frames, device):
    image_features = []
    for i in range(0, len(frames), 4):
        inputs = processor_clip(images=frames[i:i+4], return_tensors="pt").to(device)
        image_features.append(model_clip.get_image_features(**inputs).detach().cpu().numpy())
    image_features = np.concatenate(image_features, axis=0)

    cosine_sim_matrix = cosine_similarity(image_features)
    np.fill_diagonal(cosine_sim_matrix, 0)  # set diagonal elements to 0
    frame_similarity = cosine_sim_matrix.sum() / (len(frames) * (len(frames)-1))

    return frame_similarity

def TextualAlignment(PATH, device):
    with open('TextualAlignment.csv', 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'style', 'object', 'background', 'multiple']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        super_video_dirs = os.listdir(PATH)
        mean_scores_each_source = []
        for super_video_dir in super_video_dirs:
            if super_video_dir.startswith('.') or super_video_dir.startswith('_') or super_video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir)):
                continue
            video_dirs = os.listdir(os.path.join(PATH, super_video_dir))
            if len(video_dirs) == 0:
                continue
            for video_dir in video_dirs:
                if video_dir.startswith('.') or video_dir.startswith('_') or video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir)):
                    continue 
                sub_dirs = os.listdir(os.path.join(PATH,super_video_dir,video_dir))
                print("sub_dirs",sub_dirs)
                for sub_dir in sub_dirs:
                    print("sub_dir",sub_dir)
                    if sub_dir.startswith('.') or sub_dir.startswith('_') or sub_dir.startswith('config'):# or not os.path.isdir(sub_dir):
                        continue
                    vs = os.listdir(os.path.join(PATH, super_video_dir, video_dir, sub_dir))
                    print("vs",vs)
                    for v in vs:
                        if v.startswith('.') or v.startswith('_') or v.endswith('.jpg'):
                            continue
                        print(v)
                        # load gif file and decompose it into images
                        imgs = []
                        gif = Image.open(os.path.join(PATH, super_video_dir, video_dir, sub_dir, v))
                        for frame in range(gif.n_frames):
                            gif.seek(frame)
                            imgs.append(gif.copy())
                        # print("len(gif):",len(imgs))
                        # remove of final text '.gif' and get the prompt
                        score = textual_alignment_score(imgs,v[:-4],device)
                        # write into csv file
                        # check sub_dir name
                        if sub_dir == 'style':
                            writer.writerow({'video_name': v[:-4], 'style': score})
                        elif sub_dir == 'object':
                            writer.writerow({'video_name': v[:-4], 'object': score})
                        elif sub_dir == 'background':
                            writer.writerow({'video_name': v[:-4], 'background': score})
                        elif sub_dir == 'multiple':
                            writer.writerow({'video_name': v[:-4], 'multiple': score})
                        else:
                            print("error",os.path.join(PATH, super_video_dir, video_dir, sub_dir, v))
                        
                        mean_scores_each_source.append(score)
        
        # print(mean_scores_each_source)
        # print(sum(mean_scores_each_source)/len(mean_scores_each_source))
        return sum(mean_scores_each_source)/len(mean_scores_each_source)

def FrameConsistency(PATH, device):
    with open('FrameConsistency.csv', 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'style', 'object', 'background', 'multiple']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        super_video_dirs = os.listdir(PATH)
        mean_scores_each_source = []
        for super_video_dir in super_video_dirs:
            if super_video_dir.startswith('.') or super_video_dir.startswith('_') or super_video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir)):
                continue
            video_dirs = os.listdir(os.path.join(PATH, super_video_dir))
            if len(video_dirs) == 0:
                continue
            for video_dir in video_dirs:
                if video_dir.startswith('.') or video_dir.startswith('_') or video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir)):
                    continue 
                sub_dirs = os.listdir(os.path.join(PATH, super_video_dir, video_dir))
                for sub_dir in sub_dirs:
                    if sub_dir.startswith('.') or sub_dir.startswith('_') or sub_dir.startswith('config'):# or not os.path.isdir(sub_dir):
                        continue

                    vs = os.listdir(os.path.join(PATH, super_video_dir, video_dir, sub_dir))
                    for v in vs:
                        if v.startswith('.') or v.startswith('_') or v.endswith('.jpg'):
                            continue
                        # load gif file and decompose it into images
                        imgs = []
                        gif = Image.open(os.path.join(PATH, super_video_dir, video_dir, sub_dir, v))
                        for frame in range(gif.n_frames):
                            gif.seek(frame)
                            imgs.append(gif.copy())
                        score = frame_consistency_score(imgs) 
                        if sub_dir == 'style':
                            writer.writerow({'video_name': v[:-4], 'style': score})
                        elif sub_dir == 'object':
                            writer.writerow({'video_name': v[:-4], 'object': score})
                        elif sub_dir == 'background':
                            writer.writerow({'video_name': v[:-4], 'background': score})
                        elif sub_dir == 'multiple':
                            writer.writerow({'video_name': v[:-4], 'multiple': score})
                        else:
                            print("error",os.path.join(PATH, super_video_dir, video_dir, sub_dir, v))
                        
                        mean_scores_each_source.append(score)
        
        # print(mean_scores_each_source)
        # print(sum(mean_scores_each_source)/len(mean_scores_each_source))
        return sum(mean_scores_each_source)/len(mean_scores_each_source)

def PickScore(PATH, device):
    with open('PickScore.csv', 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'style', 'object', 'background', 'multiple']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        super_video_dirs = os.listdir(PATH)
        mean_scores_each_source = []
        for super_video_dir in super_video_dirs:
            if super_video_dir.startswith('.') or super_video_dir.startswith('_') or super_video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir)):
                continue
            video_dirs = os.listdir(os.path.join(PATH, super_video_dir))
            if len(video_dirs) == 0:
                continue
            for video_dir in video_dirs:
                if video_dir.startswith('.') or video_dir.startswith('_') or video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir, video_dir)):
                    continue 
                sub_dirs = os.listdir(os.path.join(PATH, super_video_dir, video_dir))
                for sub_dir in sub_dirs:
                    if sub_dir.startswith('.') or sub_dir.startswith('_') or sub_dir.startswith('config'):
                        continue

                    vs = os.listdir(os.path.join(PATH, super_video_dir, video_dir, sub_dir))
                    for v in vs:
                        if v.startswith('.') or v.startswith('_') or v.endswith('.jpg'):
                            continue
                        # load gif file and decompose it into images
                        imgs = []
                        gif = Image.open(os.path.join(PATH, super_video_dir, video_dir, sub_dir, v))
                        for frame in range(gif.n_frames):
                            gif.seek(frame)
                            imgs.append(gif.copy())
                        score = calc_probs(v[:-4], imgs)
                        # write into csv file
                        # check sub_dir name
                        if sub_dir == 'style':
                            writer.writerow({'video_name': v[:-4], 'style': score})
                        elif sub_dir == 'object':
                            writer.writerow({'video_name': v[:-4], 'object': score})
                        elif sub_dir == 'background':
                            writer.writerow({'video_name': v[:-4], 'background': score})
                        elif sub_dir == 'multiple':
                            writer.writerow({'video_name': v[:-4], 'multiple': score})
                        else:
                            print("error",os.path.join(PATH, super_video_dir, video_dir, sub_dir, v))
                        
                        mean_scores_each_source.append(score)
        
        # print(mean_scores_each_source)
        # print(sum(mean_scores_each_source)/len(mean_scores_each_source))
        return sum(mean_scores_each_source)/len(mean_scores_each_source)



def cal(PATH='/Users/jinbin/CAMP_videos',device='cuda:0'): 
    model.to(device)
    model_clip.to(device)
   




def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='/Users/jinbin/Demo_Folder/')
    args.add_argument('--device', type=str, default='cuda:0')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model.to(device)
    model_clip.to(device)

    TextualAlignmentScore = TextualAlignment(args.path, device)
    # clean gpu memory
    torch.cuda.empty_cache()
    FrameConsistencyScore = FrameConsistency(args.path, device)
    # clean gpu memory
    torch.cuda.empty_cache()
    PickScoreScore = PickScore(args.path, device)
    print("TextualAlignmentScore",TextualAlignmentScore)
    print("FrameConsistencyScore",FrameConsistencyScore)
    print("PickScoreScore",PickScoreScore)

if __name__ == '__main__':
    main()
    