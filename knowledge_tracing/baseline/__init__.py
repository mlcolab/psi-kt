# import ipdb
# ipdb.set_trace()
# import models.AKT, models.DKT, models.DKTForgetting, models.HKT, models.HLR, models.PPE

# DKT = models.DKT.DKT

# # from knowledge_tracing.baseline.BaseModel import *
# # from knowledge_tracing.baseline.models import DKT, DKTForgetting, HKT, AKT, HLR, PPE


# # __all__ = [
# #     'BaseModel', 'BaseLearnerModel',
# #     'DKT', 'DKTForgetting', 'HKT', 'AKT',
# #     'PPE', 'HLR',
# # ]

# from os.path import dirname, basename, isfile, join
# import glob

# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [
#     basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
# ]
