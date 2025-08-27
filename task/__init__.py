from .base import BaseTask
from .pii_leakage import PIILeakage
from .alpaca import AlpacaEval
from .ifeval import IFEval
from .gsm8k import GSM8k
from .gpqa import GPQA
from .spa_vl_harm import spa_vl_harm

def get_task(name) -> BaseTask:
    if "pii" in name.lower() and "leakage" in name.lower():
        return PIILeakage(name)
    elif "alpaca" in name.lower():
        return AlpacaEval(name)
    elif "ifeval" in name.lower():
        return IFEval(name)
    elif "gsm8k" in name.lower():
        return GSM8k(name)
    elif "gpqa" in name.lower():
        return GPQA(name)
    elif "spa" in name.lower() and "vl" in name.lower() and "harm" in name.lower():
        return spa_vl_harm(name)
    elif "mmsafetybench" in name.lower():
        if "local" in name.lower():
            from .mmsafetybench_local import mmsafetybench_local
            return mmsafetybench_local(name)
        else:
            from .mmsafetybench import mmsafetybench
            return mmsafetybench(name)
    elif "mmvet" in name.lower():
        from .mmvet import mmvet
        return mmvet(name)
    else:
        raise KeyError