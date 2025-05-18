from lib.generator.gfn import FMGFlowNetGenerator, TBGFlowNetGenerator
from lib.generator.diff import DiscreteDiffusion


def get_generator(args, tokenizer):
    if args.diffusion_generator :
        return DiscreteDiffusion(args, tokenizer)
    if not args.gen_do_explicit_Z:
        # 우리가 사용할 diffusion model이 아래 클래스의 method를 갖고 있으면 되는 거잖아
        return FMGFlowNetGenerator(args, tokenizer)
    else:
        return TBGFlowNetGenerator(args, tokenizer)