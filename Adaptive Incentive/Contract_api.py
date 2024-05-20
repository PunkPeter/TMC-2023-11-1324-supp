import sys, os
import time, random
import pprint
from solcx import set_solc_version

import web3, json
from web3 import Web3
# 用于编译智能合约，.sol
from solcx import compile_source

set_solc_version('v0.8.0')


# 编译合约
def compile_source_file(file_path):
    with open(file_path, 'r') as f:
        source = f.read()
    return compile_source(source)


# 部署合约
def deploy_contract(w3, contract_interface):
    account = w3.eth.accounts[0]

    contract = w3.eth.contract(
        abi=contract_interface['abi'],
        bytecode=contract_interface['bin'])
    tx_hash = contract.constructor().transact({'from': account, 'gas': 500_000_000})

    address = w3.eth.getTransactionReceipt(tx_hash)['contractAddress']
    return address
