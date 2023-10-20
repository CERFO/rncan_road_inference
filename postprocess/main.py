import argparse
import postprocess_prediction as pp
import postprocess_gte as pg
import create_masks_for_comp as cm


if __name__ == '__main__':
    print("Initiate postprocess ...")

    parser = argparse.ArgumentParser()

    # Global index annotations
    parser.add_argument('-index_path', action='store', dest='index_path', type=str,
                        help='Index file path', required=True)

    # Postprocess of the GTE
    parser.add_argument('-postprocess_gte', action='store', dest='postprocess_gte', type=bool,
                        help='Postprocess GTE should be executed', required=False, default=False)
    parser.add_argument('-gte_pred_dir', action='store', dest='gte_pred_dir', type=str,
                        help='GTE prediction directory path', required=False, default=None)
    parser.add_argument('-gte_save_dir', action='store', dest='gte_save_dir', type=str,
                        help='GTE postprocess results save directory path', required=False, default=None)

    # Postprocess of the predictions
    parser.add_argument('-postprocess_pred', action='store', dest='postprocess_pred', type=bool,
                        help='Prediction model name', required=False, default=False)
    parser.add_argument('-pred_model_name', action='store', dest='pred_model_name', type=str,
                        help='Prediction directory path', required=False, default=None)
    parser.add_argument('-pred_dir', action='store', dest='pred_dir', type=str,
                        help='Prediction postprocess results save folder name', required=False, default=None)
    parser.add_argument('-pred_save_folder', action='store', dest='pred_save_folder', type=str,
                        help='Prediction postprocess results save folder name', required=False, default=None)
    parser.add_argument('-pred_save_dir', action='store', dest='pred_save_dir', type=str,
                        help='Prediction postprocess results save directory path', required=False, default=None)

    # Create masks for comparison
    parser.add_argument('-create_mask', action='store', dest='create_mask', type=bool,
                        help='Create masks for comparison should be executed', required=False, default=False)
    parser.add_argument('-mask_model_name', action='store', dest='pred_model_name', type=str,
                        help='Model name', required=False, default=None)
    parser.add_argument('-mask_save_dir', action='store', dest='mask_save_dir', type=str,
                        help='Mask save directory path', required=False, default=None)
    parser.add_argument('-mask_gt', action='store', dest='mask_gt', type=str,
                        help='Mask ground truth directory path', required=False, default=None)
    parser.add_argument('-mask_pred_dir', action='store', dest='mask_pred_dir', type=str,
                        help='Mask prediction directory path', required=False, default=None)

    args = parser.parse_args()


    # Postprocess of the GTE
    print(f"Postprocess of the GTE: {args.postprocess_gte}")
    try:
        if args.postprocess_gte:
            if args.gte_pred_dir is None or args.gte_save_dir is None:
                raise ValueError('gte_pred_dir and gte_save_dir must be specified.')

            print(f"Execute postprocess of the GTE with {args.gte_pred_dir} to {args.gte_save_dir} ...")
            pg.execute(args.index_path, args.gte_pred_dir, args.gte_save_dir)
        else:
            print(f"Skip postprocess of the GTE")

    except ValueError as e:
        print(f"Error, exiting the postprocess GTE execution: {e}")

    # Postprocess of the predictions
    print(f"Postprocess of the predictions: {args.postprocess_pred}")
    try:
        if args.postprocess_pred:
            if (args.pred_model_name is None or args.pred_dir is None or args.pred_save_folder is None
                    or args.pred_save_dir is None):
                raise ValueError('pred_model_name, pred_dir and pred_save_folder, pred_save_dir must be specified.')

            print(f"Execute postprocess of the predictions with {args.pred_model_name}"
                  f", using {args.pred_dir} to {args.pred_save_dir} ...")
            pp.execute(args.pred_model_name, args.pred_save_folder, args.pred_dir
                       , args.pred_save_dir, args.index_path)
        else:
            print(f"Skip postprocess of the predictions")

    except ValueError as e:
        print(f"Error, exiting the postprocess predictions execution: {e}")

    # Create masks for comparison
    print(f"Create masks for comparison: {args.create_mask}")
    try:
        if args.create_mask:
            if (args.mask_model_name is None or args.mask_save_dir is None or args.mask_gt is None
                    or args.mask_pred_dir is None):
                raise ValueError('mask_model_name, mask_save_dir, mask_gt, mask_pred_dir must be specified.')

            print(f"Execute create masks for comparison with {args.save_dir} ...")
            cm.execute(
                args.index_path
                , args.mask_gt
                , args.mask_pred_dir
                , args.mask_save_dir
                , args.mask_model_name)

        else:
            print(f"Skip create masks for comparison")

    except ValueError as e:
        print(f"Error, exiting the masks generation execution: {e}")

    print("Done.")
