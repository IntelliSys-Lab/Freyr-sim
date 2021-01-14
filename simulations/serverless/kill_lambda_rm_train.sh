#! /bin/bash

ps aux | grep -v grep | grep -E "lambda_rm_train|multiprocessing.spawn" | awk {'print $2'} | xargs kill
