# Memory System for RAG Pipeline

Hệ thống memory cho RAG pipeline hỗ trợ hội thoại đa lượt (multi-turn conversation) với khả năng ghi nhớ ngữ cảnh ngắn hạn và dài hạn.

## 🏗️ Kiến trúc

```
Memory System
├── ShortTermMemory      # Lưu trữ hội thoại gần đây trong session
├── LongTermMemory       # Lưu trữ lịch sử dài hạn với vector search
└── ConversationManager  # Quản lý và merge context từ cả hai
```

## 📦 Components

### 1. ShortTermMemory

**Chức năng**: Lưu trữ các message gần đây trong session hiện tại (in-memory).

**Tính năng**:
- Window-based retrieval (N message gần nhất)
- TTL tự động cleanup
- Thread-safe operations
- Session management

**Sử dụng**:
```python
from memory import ShortTermMemory

short_term = ShortTermMemory(max_messages_per_session=20, ttl_hours=24)

# Lưu message
short_term.store_message("session_123", "Hello", "user")

# Lấy lịch sử gần đây
history = short_term.get_recent_history("session_123", window_size=5)
```

### 2. LongTermMemory

**Chức năng**: Lưu trữ lịch sử hội thoại dài hạn với vector search trong Qdrant.

**Tính năng**:
- Vector embedding và semantic search
- User profiling và context summarization
- Advanced filtering (user_id, session_id, timestamp)
- Data cleanup tự động

**Sử dụng**:
```python
from memory import LongTermMemory

long_term = LongTermMemory(collection_name="conversation_memory")

# Lưu message
long_term.store_message("session_123", "I have a headache", "user", user_id="user_1")

# Truy xuất message liên quan
related = long_term.retrieve_relevant_history("What about my medical history?", user_id="user_1")

# Tóm tắt user profile
profile = long_term.get_user_context_summary("user_1")
```

### 3. ConversationManager

**Chức năng**: Quản lý việc merge context từ short-term và long-term memory.

**Tính năng**:
- Smart context merging
- Context truncation để tránh token overflow
- Priority-based context (current query > short-term > long-term)
- Error handling với fallback mechanism

**Sử dụng**:
```python
from memory import ConversationManager

conv_manager = ConversationManager(
    max_context_tokens=2048,
    short_term_window=5,
    long_term_k=3
)

# Lấy enriched context
context = conv_manager.get_enriched_context(
    session_id="session_123",
    current_query="Should I see a doctor?",
    user_id="user_1"
)

# Cập nhật memory
conv_manager.update_memory("session_123", "Hello", "user", user_id="user_1")
```

## 🚀 Tích hợp với RAG Pipeline

### Sử dụng ConversationalLLM_RAG

```python
from inference_pipeline import ConversationalLLM_RAG

# Khởi tạo
conv_rag = ConversationalLLM_RAG(
    max_context_tokens=2048,
    short_term_window=5,
    long_term_k=3
)

# Chat với conversation context
result = conv_rag.generate(
    query="What about my medical history?",
    session_id="session_123",
    user_id="user_1",
    enable_conversation=True
)

print(result["answer"])
print(result["conversation_history"])  # Context được sử dụng
```

### Fallback Mechanism

Hệ thống tự động fallback về RAG pipeline gốc nếu:
- Memory components không khả dụng
- Có lỗi trong conversation processing
- User tắt conversation mode

## 🔧 Configuration

### Environment Variables

Thêm vào `.env`:
```bash
# Memory settings
CONVERSATION_MEMORY_COLLECTION="conversation_memory"
MAX_CONTEXT_TOKENS=2048
SHORT_TERM_WINDOW=5
LONG_TERM_K=3
```

### Settings.py

```python
# Memory config
CONVERSATION_MEMORY_COLLECTION: str = "conversation_memory"
MAX_CONTEXT_TOKENS: int = 2048
SHORT_TERM_WINDOW: int = 5
LONG_TERM_K: int = 3
```

## 🧪 Testing

Chạy test để verify tích hợp:

```bash
python test_conversation_integration.py
```

## 📊 Monitoring

### Memory Stats

```python
# Lấy thống kê memory system
stats = conv_rag.get_memory_stats()
print(stats)
```

### User Profiling

```python
# Lấy profile user
profile = conv_rag.get_user_profile("user_1")
print(profile)
```

### Session Management

```python
# Thông tin session
session_info = conv_rag.get_session_info("session_123")

# Xóa session
conv_rag.clear_session("session_123")
```

## 🚨 Error Handling

### Graceful Degradation

- Nếu Qdrant không khả dụng → chỉ sử dụng short-term memory
- Nếu embedding model lỗi → fallback về standard RAG
- Nếu memory operations fail → continue với query gốc

### Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# Memory operations sẽ log chi tiết
```

## 🔮 Future Enhancements

- [ ] Redis backend cho short-term memory
- [ ] Conversation summarization với LLM
- [ ] Advanced topic tracking
- [ ] Multi-user conversation support
- [ ] Conversation analytics và insights

## 📝 Example Use Cases

### 1. Medical Consultation

```python
# Lượt 1
result1 = conv_rag.generate("I have a headache", session_id="medical_123", user_id="patient_1")

# Lượt 2 - sử dụng context từ lượt 1
result2 = conv_rag.generate("What about my medical history?", session_id="medical_123", user_id="patient_1")
```

### 2. Customer Support

```python
# Lượt 1
result1 = conv_rag.generate("My order hasn't arrived", session_id="support_456", user_id="customer_1")

# Lượt 2 - bot nhớ về order issue
result2 = conv_rag.generate("Can you check the status?", session_id="support_456", user_id="customer_1")
```

### 3. Educational Assistant

```python
# Lượt 1
result1 = conv_rag.generate("Explain machine learning", session_id="edu_789", user_id="student_1")

# Lượt 2 - tiếp tục topic ML
result2 = conv_rag.generate("What about deep learning?", session_id="edu_789", user_id="student_1")
``` 