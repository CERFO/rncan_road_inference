import argparse
import get_sats_stats as gss
import create_dataset as cd
import clip_tiles_from_masks as ct
import create_graph_dataset as cg


if __name__ == '__main__':
    print("Initiate dataset building ...")

    parser = argparse.ArgumentParser()

    # Global parameters
    parser.add_argument('-save_dir', action='store', dest='save_dir', type=str,
                        required=True, default=None)
    parser.add_argument('-data_dir', action='store', dest='data_dir', type=str,
                        required=True, default=None)

    # Data retrieval parameters
    parser.add_argument('-stats_output_extension', action='store', dest='stats_output_extension', type=str,
                        required=False, default="_metadata.npz")
    parser.add_argument('-gs', action='store', dest='get_satellites', type=str,
                        required=False, default='geoeye-1-ortho-pansharp,worldview-2-ortho-pansharp'
                                                         ',worldview-3-ortho-pansharp,worldview-4-ortho-pansharp')
    parser.add_argument('-gb', action='store', dest='get_bands', type=str,
                        required=False, default='R,G,B,N')

    # Create dataset parameters
    parser.add_argument('-annotations_index_path', action='store', dest='annotations_index_path', type=str,
                        required=True, default=None)
    parser.add_argument('-annotations_dir', action='store', dest='annotations_dir', type=str,
                        required=True, default=None)

    # Image processing parameters
    parser.add_argument('-tile_mask_dir', action='store', dest='tile_mask_dir', type=str,
                        help='Mask directory', required=True)
    parser.add_argument('-tile_save_dir', action='store', dest='tile_save_dir', type=str,
                        help='Directory to save tiles', required=True)
    parser.add_argument('-tile_size', action='store', dest='tile_size', type=int,
                        help='Size of tiles', required=False, default=4096)
    parser.add_argument('-tile_overlap', action='store', dest='tile_overlap', type=int,
                        help='Overlap between tiles', required=False, default=256)

    # Create graph parameters
    parser.add_argument('-graph_gt', action='store', dest='graph_gt', type=str,
                        required=True)
    parser.add_argument('-graph_save_dir', action='store', dest='graph_save_dir', type=str,
                        required=True)
    parser.add_argument('-graph_image_size', action='store', dest='graph_image_size', type=int,
                        required=False, default=512)
    parser.add_argument('-graph_image_overlap', action='store', dest='graph_image_overlap', type=float,
                        required=False, default=0.33)
    parser.add_argument('-graph_encoding_max_degree', action='store', dest='graph_encoding_max_degree', type=int,
                        help='Encoding max degree', required=False, default=6)
    parser.add_argument('-graph_encoding_r', action='store', dest='graph_encoding_r', type=int,
                        required=False, default=1)
    parser.add_argument('-graph_linestring_delta_meters', action='store', dest='graph_linestring_delta_meters', type=int,
                        required=False, default=20)

    args = parser.parse_args()

    try:
        # Get dataset
        try:
            if args.data_dir is None or args.save_dir is None or args.file_name is None:
                raise ValueError('data_dir, save_dir, file_name must be specified.')

            print(f"Execute data retrieval process with {args.data_dir} to {args.save_dir} ...")
            gss.execute(args.data_dir, args.save_dir, args.stats_output_extension
                            , [item for item in args.get_satellites.split(',')]
                            , [item for item in args.get_bands.split(',')]
                        )

        except ValueError as ex:
            print(f"Error, exiting the data retrieval process execution: {ex}")
            raise ex

        # Binary masks
        try:
            if args.annotations_index_path is None or args.annotations_dir is None:
                raise ValueError('annotations_index_path and annotations_dir must be specified.')

            print(f"Execute binary masks process with {args.data_dir} to {args.save_dir} ...")
            cd.execute(args.annotations_index_path, args.annotations_dir, args.data_dir, args.save_dir)

        except ValueError as ex:
            print(f"Error, exiting the binary masks process execution: {ex}")
            raise ex

        # Image processing - clip tiles from masks
        try:
            if args.tile_mask_dir is None or args.tile_save_dir is None:
                raise ValueError('tile_mask_dir, tile_save_dir must be specified.')

            print(f"Execute binary masks process with {args.tile_mask_dir} to {args.tile_save_dir} ...")
            ct.execute(args.tile_mask_dir, args.tile_save_dir, args.tile_size, args.tile_overlap)
        except ValueError as ex:
            print(f"Error, exiting the image processing process execution: {ex}")
            raise ex

        # Create graph
        try:
            if args.graph_gt is None or args.graph_save_dir is None:
                raise ValueError('graph_gt, graph_save_dir must be specified.')

            cg.execute(args.data_dir, args.graph_gt, args.graph_save_dir
                       , [item for item in args.get_bands.split(',')], args.graph_img_size, args.graph_overlap
                       , args.graph_encoding_max_degree, args.graph_encoding_r, args.graph_linestring_delta_meters)

        except ValueError as ex:
            print(f"Error, exiting the create graph process execution: {ex}")
            raise ex

    except ValueError as ex:
        print(f"Exiting")
        exit(-1)

    print("Done.")
