"""DBConformer network factory used by the LOSO experiment."""

from models.DBConformer import DBConformer
from models.TriViewDBConformer import TriViewDBConformer
from models.TriViewConcatDBConformer import TriViewConcatDBConformer


def backbone_net_dbconformer(args):
    """Build the DBConformer model from the LOSO experiment arguments."""
    return DBConformer(
        args,
        emb_size=args.emb_size,
        tem_depth=args.transformer_depth_tem,
        chn_depth=args.transformer_depth_chn,
        chn=args.chn,
        n_classes=args.class_num,
    )


def backbone_net_triview_dbconformer(args):
    """Build the tri-view DBConformer without changing the original factory."""
    return TriViewDBConformer(
        args,
        emb_size=args.emb_size,
        tem_depth=args.transformer_depth_tem,
        chn_depth=args.transformer_depth_chn,
        chn=args.chn,
        n_classes=args.class_num,
    )


def backbone_net_triview_concat_dbconformer(args):
    """Build the 120-D weighted-concatenation tri-view DBConformer."""
    return TriViewConcatDBConformer(
        args,
        emb_size=args.emb_size,
        tem_depth=args.transformer_depth_tem,
        chn_depth=args.transformer_depth_chn,
        chn=args.chn,
        n_classes=args.class_num,
    )
