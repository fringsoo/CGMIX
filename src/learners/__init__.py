from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner

from .dcg_learner import DCGLearner
REGISTRY["dcg_learner"] = DCGLearner

from .cgmix_learner import CgmixLearner
REGISTRY['cgmix_learner'] = CgmixLearner