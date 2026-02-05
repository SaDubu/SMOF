#pragma once
#include "define.h"
// Single Producer single consumer
template<typename T>
class LockFreeQueueSPSC
{
private:
	struct Node
	{
		std::shared_ptr<T> Data;
		Node* Next;
		Node() : Next(nullptr)
		{}
	};

	std::atomic<Node*> Head;
	std::atomic<Node*> Tail;

	Node* PopHead()
	{
		Node* const OldHead = Head.load();
		if (OldHead == Tail.load())
		{
			return nullptr;
		}
		Head.store(OldHead->Next);
		return OldHead;
	}

public:
	LockFreeQueueSPSC() : Head(new Node), Tail(Head.load()) {}
	~LockFreeQueueSPSC()
	{
		while (Node* const OldHead = Head.load())
		{
			Head = OldHead->Next;
			delete OldHead;
		}
	}
	std::shared_ptr<T> Pop()
	{
		Node* OldHead = PopHead();
		if (!OldHead)
		{
			return std::shared_ptr<T>();
		}
		std::shared_ptr<T> const Res(OldHead->Data);
		delete OldHead;
		return Res;
	}
	void Push(const T& NewValue)
	{
		std::shared_ptr<T> NewData(std::make_shared<T>(NewValue));
		Node* p = new Node;
		Node* const OldTail = Tail.load();
		OldTail->Data.swap(NewData);
		OldTail->Next = p;
		Tail.store(p);
	}
};

template<typename T>
class LockFreeQueue
{
private:
	struct Node;

	// 외부로 노출되는 포인터
	struct CountedNodePtr
	{
		int ExternalCount;
		Node* Ptr;
	};
	
	std::atomic<CountedNodePtr> Head;
	std::atomic<CountedNodePtr> Tail;

	struct NodeCounter
	{
		unsigned InternalCount : 30;   // use 30 bits
		unsigned ExternalCounters : 2; // use 2 bits (최대 2개를 가지므로 2비트로 표현 가능)
	};

	struct Node
	{
		std::atomic<T*> Data;
		std::atomic<NodeCounter> Count; // Internal Count + External Counters
		std::atomic<CountedNodePtr> Next;

		Node()
			:
			Next(CountedNodePtr({}))
		{
			NodeCounter NewCount;
			NewCount.InternalCount = 0;
			NewCount.ExternalCounters = 2; // Tail Node와 Tail의 Previous Node의 Next
			Count.store(NewCount);
		}

		void ReleaseRef()
		{
			NodeCounter OldCounter = Count.load(std::memory_order_relaxed);
			NodeCounter NewCounter;
			// Count -> NewCounter(내부 참조가 하나 감소한)
			do
			{
				NewCounter = OldCounter;
				--NewCounter.InternalCount;
			} while (!Count.compare_exchange_strong(OldCounter, NewCounter, std::memory_order_acquire, std::memory_order_relaxed));

			// 내부 참조 0, 외부 카운터 0이면 삭제
			if (NewCounter.InternalCount == 0 && NewCounter.ExternalCounters == 0)
			{
				delete this;
			}
		}
	};

public:
	LockFreeQueue()
	{
		CountedNodePtr NewNode;
		NewNode.Ptr = new Node();
		NewNode.ExternalCount = 0;
		Head.store(NewNode);
		Tail = Head.load();
	}
	~LockFreeQueue()
	{
		CountedNodePtr Cur = Head.load();
		while (Cur.Ptr)
		{
			Node* Temp = Cur.Ptr;
			Cur = Cur.Ptr->Next;
			delete Temp;
		}
	}
	size_t Size() const
	{
		size_t Size = 0;
		CountedNodePtr Cur = Head.load();
		while (Cur.Ptr)
		{
			++Size;
			Cur = Cur.Ptr->Next;
		}
		assert(Size > 0);
		return Size - 1;
	}
private:
	static void IncreaseExternalCount(std::atomic<CountedNodePtr>& Counter, CountedNodePtr& OldCounter)
	{
		CountedNodePtr NewCounter;
		do
		{
			NewCounter = OldCounter;
			++NewCounter.ExternalCount;
		} while (!Counter.compare_exchange_strong(OldCounter, NewCounter, std::memory_order_acquire, std::memory_order_relaxed));

		OldCounter.ExternalCount = NewCounter.ExternalCount;
	}

	static void FreeExternalCounter(CountedNodePtr& OldNodePtr)
	{
		Node* const Ptr = OldNodePtr.Ptr;
		const int CountIncrease = OldNodePtr.ExternalCount - 2;
		NodeCounter OldCounter = Ptr->Count.load(std::memory_order_relaxed);

		NodeCounter NewCounter;
		do
		{
			NewCounter = OldCounter;
			--NewCounter.ExternalCounters; // 외부 참조 카운터 삭제
			NewCounter.InternalCount += CountIncrease; // 해당 내부 참조에 순 외부 참조 증가 수 만큼 더해준다.
		} while (!Ptr->Count.compare_exchange_strong(OldCounter, NewCounter, std::memory_order_acquire, std::memory_order_relaxed));

		// 삭제 조건
		if (NewCounter.InternalCount == 0 && NewCounter.ExternalCounters == 0)
		{
			delete Ptr;
		}
	}

public:
	std::unique_ptr<T> Pop()
	{
		CountedNodePtr OldHead = Head.load(std::memory_order_relaxed);

		for (;;)
		{
			IncreaseExternalCount(Head, OldHead);

			Node* const Ptr = OldHead.Ptr;

			// 큐가 비어있음
			if (Ptr == Tail.load().Ptr)
			{
				Ptr->ReleaseRef();
				return std::unique_ptr<T>();
			}

			CountedNodePtr Next = Ptr->Next.load();
			if (Head.compare_exchange_strong(OldHead, Next))
			{
				T* const Res = Ptr->Data.exchange(nullptr);
				FreeExternalCounter(OldHead);
				return std::unique_ptr<T>(Res);
			}

			Ptr->ReleaseRef();
		}
	}

private:
	// (1) 실제 Tail노드가 업데이트 되는 함수
	void SetNewTail(CountedNodePtr& OldTail, CountedNodePtr const& NewTail)
	{
		Node* const CurrentTailPtr = OldTail.Ptr;

		// (2) 다른 쓰레드에 의해 Tail이 바뀌지 않았을 때만 loop를 돈다.
		while (!Tail.compare_exchange_weak(OldTail, NewTail)
			&& OldTail.Ptr == CurrentTailPtr);

		// (3) 현재 쓰레드가 Tail을 업데이트했을 때
		if (OldTail.Ptr == CurrentTailPtr)
		{
			// (4) 해당 OldTail의 외부 참조 카운터를 제거한다.
			FreeExternalCounter(OldTail);
		}
		else
		{
			// (5) 다른 쓰레드가 Tail을 바꾸었으므로 현재 OldTail은 읽기만 종료한 것이다.
			CurrentTailPtr->ReleaseRef();
		}
	}

public:
	void Push(const T& NewValue)
	{
		std::unique_ptr<T> NewData(new T(NewValue));
		CountedNodePtr NewNext;
		NewNext.Ptr = new Node;
		NewNext.ExternalCount = 1;
		CountedNodePtr OldTail = Tail.load();

		for (;;)
		{
			IncreaseExternalCount(Tail, OldTail);

			// (6) 데이터 삽입을 시도한다.
			T* OldData = nullptr;
			if (OldTail.Ptr->Data.compare_exchange_strong(OldData, NewData.get()))
			{
				// (7) OldTail->Next를 자신이 생성한 노드 NewNext로 바꾸는 것을 시도한다.
				CountedNodePtr OldNext = {}; // 다음은 비어있어야한다.
				if (!OldTail.Ptr->Next.compare_exchange_strong(OldNext, NewNext))
				{
					// (8) 다른 쓰레드에 의해 노드가 삽입되었으므로 현재 쓰레드에서 노드를 삽입하지 않는다.
					delete NewNext.Ptr;
					NewNext = OldNext; // 다른 쓰레드에 의해 설정된 OldTail->Next 포인터를 사용
				}

				// (9) Tail을 업데이트한다.
				SetNewTail(OldTail, NewNext);
				NewData.release(); // 데이터 삽입은 성공했다.
				break;
			}
			else
			{
				// (10) 지금 OldTail에 데이터는 들어가 있는 상태이다.
				CountedNodePtr OldNext = {};

				// Tail 노드가 업데이트가 안되어 있을 수 있으니 다른 쓰레드에서 도와주자
				if (OldTail.Ptr->Next.compare_exchange_strong(OldNext, NewNext))
				{
					// (11) 다른 쓰레드에서 OldTail->Next를 바꾸었다.
					OldNext = NewNext;
					NewNext.Ptr = new Node; // 현재 쓰레드가 노드를 삽입한 것은 아니므로 새로운 노드를 할당한다.
				}

				// (12) Tail을 업데이트한다.
				SetNewTail(OldTail, OldNext);
			}
		} // End for
	} // End Push

public:
	void PushNotLockFree(const T& NewValue)
	{
		std::unique_ptr<T> NewData(new T(NewValue));
		CountedNodePtr NewNext;
		NewNext.Ptr = new Node;
		NewNext.ExternalCount = 1;

		CountedNodePtr OldTail = Tail.load();
		for (;;)
		{
			// OldTail에 의해 외부 참조가 하나 증가하였다.
			IncreaseExternalCount(Tail, OldTail);

			// 데이터 삽입 성공
			T* OldData = nullptr;
			if (OldTail.Ptr->Data.compare_exchange_strong(OldData, NewData.get()))
			{
				OldTail.Ptr->Next = NewNext;
				OldTail = Tail.exchange(NewNext);

				// 외부 참조 카운터의 수를 하나 줄인다
				FreeExternalCounter(OldTail);
				NewData.release();
				break;
			}

			// 외부 참조 읽기 종료 : 내부 참조 감소
			OldTail.Ptr->ReleaseRef();
		}
	}
public:
	void Push_Broken(const T& NewValue)
	{
		// Data를 compare_exchange_strong()으로 변경해야함
		// std::unique_ptr로 만들고 성공했을 때 std::atomic<T*> Data로 관리
		std::unique_ptr<T> NewData(new T(NewValue));
		CountedNodePtr NewNext;
		NewNext.Ptr = new Node;
		NewNext.ExternalCount = 1; // Tail에 의해

		for (;;)
		{
			// 1. Tail 포인터 로드
			Node* const OldTail = Tail.load();

			// 2. OldTail 역참조하여 Data가 nullptr인지 CAS
			// 3. 다른 쓰레드에서 Pop() 인해 2의 OldTail이 댕글링 포인터가 된다.
			/*  큐에 데이터가 하나 존재 (Node1(Data), Node2(null))
				스레드 1 : OldTail은 Node2
				스레드 2 : Push 또다른 노드 (Node1(Data), Node2(Data2), Node3(null))
				스레드 3 : Pop(), Pop()   (Head == Tail == Node3(null))
				스레드 1 : OldTail은 삭제된 포인터(Node2)를 가리키고 있음
			*/
			T* OldData = nullptr;
			if (OldTail->Data.compare_exchange_string(OldData, NewData.get())) // 데이터 삽입
			{
				// 큐 데이터 구조에서
				// 이전 노드의 Next로 인해 NewNext의 외부 참조가 하나 더 생긴다.
				OldTail->Next = NewNext;

				Tail.store(NewNext.Ptr);
				NewData.release(); // 메모리 릴리즈
				break;
			}
		}
	}
};