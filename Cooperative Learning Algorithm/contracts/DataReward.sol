pragma solidity ^0.8.19;

// Declare the contract name
contract DataReward {

    // Declare the contract owner's address
    address public owner;

    // Declare the contract balance
    uint256 public balance;

    // Declare a mapping of node addresses to data sizes
    mapping(address => uint256) public data;

    // Declare the reward rate, how many wei per byte
    uint256 public rewardRate = 10 wei;

    // Constructor function, sets the contract owner to the deployer
    constructor() {
        owner = msg.sender;
    }

    // Receive function, increases the contract balance
    receive() external payable {
        balance += msg.value;
    }

    // Send data function, specifying data size (in bytes)
    function sendData(uint256 _size) external {
        // Check if the data size is greater than 0
        require(_size > 0, "Data size must be positive");
        // Calculate the reward amount, which should not exceed the contract's balance
        uint256 reward = _size * rewardRate;
        require(reward <= balance, "Insufficient balance");
        // Update the mapping of node addresses and data sizes
        data[msg.sender] += _size;
        // Deduct the reward amount from the contract's balance and transfer it to the node's address
        balance -= reward;
        payable(msg.sender).transfer(reward);
    }

    // Query data function, returns the total data size (in bytes) sent by a node's address
    function queryData(address _node) external view returns (uint256) {
        return data[_node];
    }

}
