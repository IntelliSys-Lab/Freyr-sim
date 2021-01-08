#! /bin/bash

ps aux | grep -v grep | grep lambda_rm_train | awk {'print $2'} | xargs kill
