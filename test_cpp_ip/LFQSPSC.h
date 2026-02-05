#ifndef LFQSPC_H
#define LFQSPC_H

#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueueSPSC {
private:
    struct Node {
        std::unique_ptr<T> Data;
        std::atomic<Node*> Next;
        Node() : Next(nullptr) {}
    };

    alignas(64) std::atomic<Node*> Head;
    alignas(64) std::atomic<Node*> Tail;

public:
    LockFreeQueueSPSC() {
        Node* dummy = new Node;
        Head.store(dummy);
        Tail.store(dummy);
    }

    ~LockFreeQueueSPSC() {
        while (Node* oldHead = Head.load()) {
            Head.store(oldHead->Next.load(std::memory_order_relaxed));
            delete oldHead;
        }
    }

    void Push(T newValue) {
        std::unique_ptr<T> data = std::make_unique<T>(std::move(newValue));
        Node* nextNode = new Node;
        
        Node* const oldTail = Tail.load(std::memory_order_relaxed);
        oldTail->Data = std::move(data);
        
        oldTail->Next.store(nextNode, std::memory_order_release);
        Tail.store(nextNode, std::memory_order_relaxed);
    }

    bool Pop(T& result) {
        Node* const oldHead = Head.load(std::memory_order_relaxed);

        Node* const nextNode = oldHead->Next.load(std::memory_order_acquire);

        if (!nextNode) {
            return false; // Queue Empty
        }

        if (oldHead->Data) {
            result = std::move(*(oldHead->Data));
        }
        Head.store(nextNode, std::memory_order_relaxed);
        delete oldHead;
        return true;
    }
};

#endif