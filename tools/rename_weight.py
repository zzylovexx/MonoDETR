from argparse import ArgumentParser
from collections import OrderedDict

import torch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='The checkpoint to be modified')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='The file to save the modified checkpoint')
    parser.add_argument('--drop-mean', action='store_true', default=True)
    args = parser.parse_args()

    ckpt = torch.load(args.input)
    weight = ckpt['model_state']

    weight_renamed = OrderedDict()
    for key, val in weight.items():
        if key.startswith('depth_predictor.classifier'):
            weight_renamed[key.replace('depth_predictor.classifier', 'depth_predictor.depth_classifier')] = val

        elif key.startswith('depth_predictor.depth_embed'):
            weight_renamed[key.replace('depth_predictor.depth_embed', 'depth_predictor.depth_pos_embed')] = val

        elif key.startswith('depth_predictor.encoder_proj'):
            weight_renamed[key.replace('depth_predictor.encoder_proj', 'depth_predictor.depth_encoder')] = val

        elif key.startswith('transformer.'):
            weight_renamed[key.replace('transformer.', 'depthaware_transformer.')] = val
    
        elif (not args.drop_mean) or key != 'mean_size':
            weight_renamed[key] = val

    torch.save({
        **ckpt,
        'model_state': weight_renamed
    }, args.output)
    