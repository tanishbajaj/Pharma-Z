// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title AuditTrail
 * @notice Stores tamper-proof audit hashes for the Pharma-Z forensic audit system.
 * @dev Only stores hashes and metadata — NO raw data is stored on-chain.
 */
contract AuditTrail {
    
    struct AuditRecord {
        string dataHash;       // SHA-256 hash of the audited data
        string eventType;      // Type of event (e.g., "prediction", "training")
        uint256 timestamp;     // Block timestamp when stored
        address submittedBy;   // Address that submitted the record
        bool exists;           // Existence flag for verification
    }

    // Record ID => AuditRecord
    mapping(uint256 => AuditRecord) public records;
    
    // Total number of records
    uint256 public recordCount;

    // Owner of the contract
    address public owner;

    // --- Events ---
    event RecordAdded(
        uint256 indexed recordId,
        string dataHash,
        string eventType,
        uint256 timestamp,
        address submittedBy
    );

    event RecordVerified(
        uint256 indexed recordId,
        bool isValid,
        string storedHash,
        string providedHash
    );

    // --- Modifiers ---
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    // --- Constructor ---
    constructor() {
        owner = msg.sender;
        recordCount = 0;
    }

    /**
     * @notice Add a new audit record to the blockchain.
     * @param _dataHash  SHA-256 hash of the data being audited.
     * @param _eventType Type of audit event.
     * @return recordId  The ID of the newly created record.
     */
    function addRecord(
        string memory _dataHash,
        string memory _eventType
    ) public returns (uint256 recordId) {
        recordId = recordCount;
        
        records[recordId] = AuditRecord({
            dataHash: _dataHash,
            eventType: _eventType,
            timestamp: block.timestamp,
            submittedBy: msg.sender,
            exists: true
        });

        recordCount++;

        emit RecordAdded(recordId, _dataHash, _eventType, block.timestamp, msg.sender);
        
        return recordId;
    }

    /**
     * @notice Get an audit record by its ID.
     * @param _recordId The record ID to retrieve.
     * @return dataHash, eventType, timestamp, submittedBy
     */
    function getRecord(uint256 _recordId) public view returns (
        string memory dataHash,
        string memory eventType,
        uint256 timestamp,
        address submittedBy
    ) {
        require(records[_recordId].exists, "Record does not exist");
        
        AuditRecord memory record = records[_recordId];
        return (record.dataHash, record.eventType, record.timestamp, record.submittedBy);
    }

    /**
     * @notice Verify a data hash against a stored record.
     * @param _recordId    The record ID to verify against.
     * @param _providedHash The hash to compare.
     * @return isValid True if the hashes match.
     */
    function verifyRecord(
        uint256 _recordId,
        string memory _providedHash
    ) public returns (bool isValid) {
        require(records[_recordId].exists, "Record does not exist");
        
        string memory storedHash = records[_recordId].dataHash;
        isValid = keccak256(bytes(storedHash)) == keccak256(bytes(_providedHash));

        emit RecordVerified(_recordId, isValid, storedHash, _providedHash);
        
        return isValid;
    }

    /**
     * @notice Get the total number of audit records.
     * @return count Total records stored.
     */
    function getRecordCount() public view returns (uint256 count) {
        return recordCount;
    }
}
