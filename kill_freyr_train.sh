#! /bin/bash

ps aux | grep -v grep | grep -E "freyr_train|multiprocessing.spawn" | awk {'print $2'} | xargs kill
