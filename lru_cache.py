# -------------------------------------------------------------- Practice session 1 ------------------------------------------------------------------
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_head(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_head(node)
            return node.value
        return -1
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            node.value = value
            self._add_to_head(node)
            return
        
        new_node = Node(key, value)
        self._add_to_head(new_node)
        self.cache[key] = new_node

        if len(self.cache) > self.capacity:
            lru_node = self.tail.prev
            self._remove(lru_node)
            del self.cache[lru_node.key]
            


# ----------------------------------------------------------------- Practice session 2 ----------------------------------------------------------------

class Node(object):
    def __init__(self, results):
        self.results = results
        self.prev = None
        self.next = None


class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def move_to_front(self, node):
        return
    
    def append_to_front(self, node):
        return
    
    def remove_from_tail(self):
        return


class Cache(object):

    def __init__(self, MAX_SIZE):
        self.MAX_SIZE = MAX_SIZE
        self.size = 0
        self.lookup = {}
        self.linked_list = LinkedList()

    def get(self, query):
        """
        Get the stored query result from the cache.
        
        Accessing a node updates its position to the front of the LRU list.
        """
        node = self.lookup.get(query)
        if node is None:
            return None
        self.linked_list.move_to_front(node)
        return node.results
    
    def set(self, results, query):
        """
        Set the result for the given query key in the cache.

        When updating an entry, updates ints position to the front of the LRU list.
        If the entry is new and the cache is at capacity, remove the oldest entry before the new entry is added.
        """

        node = self.lookup.get(query)
        if node is not None:
            # Key exists in cache, update the value
            node.results = results
            self.linked_list.omove_to_front(node)
        else:
            # Key does not exist in cache
            if self.size == self.MAX_SIZE:
                # Remove the oldest entry from the linked list and lookup
                self.lookup.pop(self.linked_list.tail.query, None)
                self.linked_list.remove_from_tail()
            else:
                self.size += 1
            # Add the new key and value
            new_node = Node(results)
            self.linked_list.append_to_front(new_node)
            self.lookup[query] = new_node


# ----------------------------------------------------------------- Practice session 3 ----------------------------------------------------------------

class Node(object):
    def __init__(self, query, results):
        self.results = results
        self.query = query
        self.prev = None
        self.next = None

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def move_to_front(self, node):
        if not node or node == self.head:
            return

        # 1. Detach node from current position
        if node == self.tail:
            self.tail = node.prev
            self.tail.next = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev

        # 2. Attach node to the front
        node.next = self.head
        node.prev = None
        
        if self.head:
            self.head.prev = node
            
        self.head = node

    def append_to_front(self, node):
        if not self.head:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            node.prev = None

    def remove_from_tail(self):
        if not self.tail:
            return None
            
        removed_node = self.tail
        
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
            
        return removed_node

class Cache(object):
    def __init__(self, MAX_SIZE):
        self.MAX_SIZE = MAX_SIZE
        self.size = 0
        self.lookup = {}
        self.linked_list = LinkedList()

    def get(self, query):
        """
        Get the stored query result from the cache.
        Accessing a node updates its position to the front of the LRU list.
        """
        node = self.lookup.get(query)
        if node is None:
            return None
        self.linked_list.move_to_front(node)
        return node.results
    
    def set(self, results, query):
        """
        Set the result for the given query key in the cache.
        """
        node = self.lookup.get(query)
        if node is not None:
            # Key exists in cache, update value and move to front
            node.results = results
            self.linked_list.move_to_front(node) # FIXED TYPO
        else:
            # Key does not exist
            if self.size == self.MAX_SIZE:
                # Remove oldest entry
                removed_node = self.linked_list.remove_from_tail()
                if removed_node:
                    del self.lookup[removed_node.query] # Uses the stored key
            else:
                self.size += 1
            
            # Add new key and value
            new_node = Node(query, results) # Pass query here
            self.linked_list.append_to_front(new_node)
            self.lookup[query] = new_node


# -------------------------------------------------------------- Practice session 4 ------------------------------------------------------------------

class Node(object):
    def __init(self, query, results):
        self.results = results
        self.query = query
        self.prev = None
        self.next = None

class LinkedList(object):
    def __init__(self):
        self.head = None
        self.tail = None

    def move_to_front(self, node):
        if not node or node == self.head:
            return
        
        if node == self.tail:
            self.tail = node.prev
            self.tail.next = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev

        node.next = self.head
        node.prev = None

        if self.head:
            self.head.prev = node

        self.head = node

    def append_to_front(self, node):
        if not self.head:
            self.head = node
            self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
            node.prev = None

    def remove_from_tail(self):
        if not self.tail:
            return None
        
        removed_node = self.tail

        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None

        return removed_node
    
class Cache(object):
    def __init__(self, MAX_SIZE):
        self.MAX_SIZE = MAX_SIZE
        self.size = 0
        self.lookup = {}
        self.linked_list = LinkedList()

    def get(self, query):
        """
        Get the stored query result from the cache.
        Accessing a node updates its position to the front of the LRU list.
        """
        node = self.lookup.get(query)
        if node is None:
            return None
        self.linked_list.move_to_front(node)
        return node.results
    
    def set(self, results, query):
        """
        Set the result for the given query key in the cache
        """
        node = self.lookup.get(query)
        if node is not None:
            node.results = results
            self.likned_list.move_to_front(node)
        else:
            if self.size == self.MAX_SIZE:
                removed_node = self.linked_list.remove_from_tail()
                if removed_node:
                    del self.lookup[removed_node.query]
            else:
                self.size += 1

            new_node = Node(query, results)
            self.linked_list.append_to_front(new_node)
            self.lookup[query] = new_node