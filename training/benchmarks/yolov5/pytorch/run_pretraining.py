"""Pytorch Pretraining Example"""
"""
说明：文档中所有TODO的地方，都需要自定义实现。尽量保证接口一致。没有标记TODO的地方，可以参考示例中的实现，或者在此基础上做些微调。
"""

# 标准库
import os
import sys
import time
from typing import Any, Tuple

# 三方库

# benchmarks目录 append到sys.path
CURR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(CURR_PATH,
                                             "../../")))  # benchmarks目录

# 本地库
import config
from driver import Event, dist_pytorch
from driver.helper import InitHelper

# TODO 导入相关的模块、方法、变量。这里保持名称一致，实现可以不同。
# from train import trainer_adapter
# from train.evaluator import Evaluator
# from train.trainer import main
from train import trainer
from train.training_state import TrainingState
# TODO 这里需要导入dataset, dataloader的相关方法。 这里尽量保证函数的接口一致，实现可以不同。
# from utils.dataloader import build_train_dataset, \
#     build_eval_dataset, build_train_dataloader, build_eval_dataloader

logger = None


def main() -> Tuple[Any, Any]:
    global logger
    global config
    
    # init
    init_helper = InitHelper(config)
    model_driver = init_helper.init_driver(globals(), locals())  # _base.py增加模型名称name
    config = model_driver.config
    dist_pytorch.init_dist_training_env(config)
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.INIT_START)

    # logger
    logger = model_driver.logger
    init_start_time = logger.previous_log_time # init起始时间，单位ms

    # TODO 得到seed
    """
    这里获取seed的可行方式：
    1. 配置文件中的seed
    2. 自定义seed的生成方式：dist_pytorch.setup_seeds得到work_seeds数组，取其中某些元素。参考GLM-Pytorch的run_pretraining.py的seed生成方式
    3. 其他自定义方式
    """

    init_helper.set_seed(config.seed, config.vendor)

    # TODO  构建dataset, dataloader 【train && validate】
    # train_dataset = build_train_dataset()
    # eval_dataset = build_eval_dataset()
    # train_dataloader = build_train_dataloader()
    # eval_dataloader = build_eval_dataloader()

    # 根据 eval_dataloader 构建evaluator
    # evaluator = Evaluator(config, eval_dataloader)

    # 创建TrainingState对象
    training_state = TrainingState()

    # 构建 trainer：依赖 evaluator、TrainingState对象
    # trainer = Trainer(driver=model_driver,
    #                   adapter=trainer_adapter,
    #                   evaluator=evaluator,
    #                   training_state=training_state,
    #                   device=config.device,
    #                   config=config)
    # training_state._trainer = trainer
    

    # 设置分布式环境, trainer init()
    # dist_pytorch.barrier(config.vendor)
    # trainer.init()
    # dist_pytorch.barrier(config.vendor)

    # evaluation统计
    # init_evaluation_start = time.time() # evaluation起始时间，单位为秒
    """
    TODO 实现Evaluator 类的evaluate()方法，用于返回关键指标信息，如loss，eval_embedding_average等。
    例如：training_state.eval_avg_loss, training_state.eval_embedding_average = evaluator.evaluate(trainer)
    """

    # init_evaluation_end = time.time() # evaluation结束时间，单位为秒
    """
    TODO 收集eval关键信息，用于日志输出
    例如： init_evaluation_info = dict(
        eval_loss=training_state.eval_avg_loss,
        eval_embedding_average=training_state.eval_embedding_average,
        time=init_evaluation_end - init_evaluation_start)
    """
    # # time单位为秒
    # init_evaluation_info = dict(time=init_evaluation_end -
    #                             init_evaluation_start)
    # model_driver.event(Event.INIT_EVALUATION, init_evaluation_info)

    # do evaluation
    if not config.do_train:
        return config, training_state

    # init 统计
    model_driver.event(Event.INIT_END)
    init_end_time = logger.previous_log_time # init结束时间，单位为ms
    training_state.init_time = (init_end_time - init_start_time) / 1e+3 # 初始化时长，单位为秒

    # TRAIN_START
    dist_pytorch.barrier(config.vendor)
    model_driver.event(Event.TRAIN_START)
    raw_train_start_time = logger.previous_log_time # 训练起始时间，单位为ms

    # 训练过程
    print("=============train================")
    
    trainer.run(config, training_state)
    
    # TRAIN_END事件
    model_driver.event(Event.TRAIN_END)
    raw_train_end_time = logger.previous_log_time # 训练结束时间，单位为ms

    # 训练时长，单位为秒
    training_state.raw_train_time = (raw_train_end_time -
                                     raw_train_start_time) / 1e+3

    return config, training_state


if __name__ == "__main__":
    start = time.time()
    config_update, state = main()
    if not dist_pytorch.is_main_process():
        sys.exit(0)
    
    # 训练信息写日志
    e2e_time = time.time() - start
    if config_update.do_train:
        
        # TODO 构建训练所需的统计信息，包括不限于：e2e_time、training_sequences_per_second、
        # converged、final_accuracy、raw_train_time、init_time              
        training_perf = (dist_pytorch.global_batch_size(config_update) *
                         state.global_steps) / state.raw_train_time
        finished_info = {
            "e2e_time": e2e_time,
            "training_sequences_per_second": training_perf,
            "converged": state.converged,
            "final_P": state.P,
            "final_R": state.R,
            "final_mAP50": state.mAP50,
            "final_mAP": state.mAP,
            "final_fitness": state.best_fitness,   
            "raw_train_time": state.raw_train_time,
            "init_time": state.init_time,
        }
    else:
        finished_info = {"e2e_time": e2e_time}
    logger.log(Event.FINISHED, message=finished_info, stacklevel=0)
