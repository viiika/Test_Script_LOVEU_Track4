import os
from PIL import Image
import pandas as pd
import imageio
import argparse


def jpg2gif(PATH):
    super_video_dirs = os.listdir(PATH)

    df = pd.read_csv(os.path.join(PATH, 'LOVEU-TGVE-2023_Dataset.csv'))
    df_prompt = df[['Video name', 'Style Change Caption', 'Object Change Caption',
                    'Background Change Caption', 'Multiple Changes Caption']].loc[1:, :]
    df_prompt['style'] = df_prompt['Style Change Caption']
    df_prompt['object'] = df_prompt['Object Change Caption']
    df_prompt['background'] = df_prompt['Background Change Caption']
    df_prompt['multiple'] = df_prompt['Multiple Changes Caption']
    # print(df_prompt)
    df_prompt = df_prompt.set_index('Video name')

    for super_video_dir in super_video_dirs:
        if super_video_dir.startswith('.') or super_video_dir.startswith('_') or super_video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir)):
            continue
        video_dirs = os.listdir(os.path.join(PATH, super_video_dir))
        for video_dir in video_dirs:
            if video_dir.startswith('.') or video_dir.startswith('_') or video_dir.startswith('cal.py') or not os.path.isdir(os.path.join(PATH, super_video_dir, video_dir)):
                continue
            sub_dirs = os.listdir(os.path.join(
                PATH, super_video_dir, video_dir))
            for sub_dir in sub_dirs:
                if sub_dir.startswith('.') or sub_dir.startswith('_') or sub_dir.startswith('config'):
                    continue
                vs = os.listdir(os.path.join(
                    PATH, super_video_dir, video_dir, sub_dir))
                for v in vs:
                    if v.startswith('.') or v.startswith('_'):
                        continue
                    if v.endswith('.gif'):
                        vs.remove(v)
                vs.sort()
                prompt = df_prompt.loc[video_dir.lower(), sub_dir.lower()]
                # concat vs as a gif file, name it as prompt
                gif_list = []
                for v in vs:
                    if v.startswith('.') or v.startswith('_'):
                        continue
                    # load jpg and png file and compose it into gif
                    img = Image.open(os.path.join(
                        PATH, super_video_dir, video_dir, sub_dir, v))
                    gif_list.append(img)
                imageio.mimsave(os.path.join(
                    PATH, super_video_dir, video_dir, sub_dir, prompt + '.gif'), gif_list, duration=0.5)


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='/Users/jinbin/Demo_Folder/')
    jpg2gif(args.path)


if __name__ == '__main__':
    main()
