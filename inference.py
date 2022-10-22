from JoJoGAN import Infer_JoJoGAN
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sample", type=int, default=100)
    parser.add_argument("--stylize_samples", type=str, default="stylize_samples")
    parser.add_argument("--stylize_model", type=str, default="ckpts/stylize_stylegan.pth")
    parser.add_argument("--psp_model", type=str, default="saved_models/e4e_ffhq_encode.pt")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if not os.path.exists(args.stylize_samples):
        os.mkdir(args.stylize_samples)
    jojogan = Infer_JoJoGAN(stylize_model=args.stylize_model, 
                            psp_model=args.psp_model, 
                            device=args.device)
    jojogan.inference(num_sample=args.num_sample, save_dir=args.stylize_samples)


