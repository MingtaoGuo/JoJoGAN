from JoJoGAN import Train_JoJoGAN
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles_dir", type=str, default="./styles1")
    parser.add_argument("--num_itr", type=int, default=500)
    parser.add_argument("--style_mix", type=int, default=7)
    parser.add_argument("--results", type=str, default="./results")
    parser.add_argument("--ckpts", type=str, default="./ckpts")
    parser.add_argument("--psp_model", type=str, default="saved_models/e4e_ffhq_encode.pt")
    parser.add_argument("--stylegan2_model", type=str, default="saved_models/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--face_detector", type=str, default="saved_models/yolov5s-face.onnx")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if not os.path.exists(args.results):
        os.mkdir(args.results)
    if not os.path.exists(args.ckpts):
        os.mkdir(args.ckpts)
    
    jojogan = Train_JoJoGAN(stylegan2_model=args.stylegan2_model, 
                            psp_model=args.psp_model, 
                            yolo_model=args.face_detector, 
                            device=args.device)
    jojogan.train(styles_dir=args.styles_dir, save_img_dir=args.results, save_model_dir=args.ckpts, num_itr=args.num_itr, style_mix_idx=args.style_mix)


