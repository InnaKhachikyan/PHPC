#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define LIST_SIZE 100

struct Node {
	int data;
	struct Node *next;
	struct Node *prev;
};

void incOmp(struct Node *head);
void incOmpTask(struct Node *head);


struct Node* createList(int size) {
    if (size <= 0) return NULL; 

    struct Node *head = malloc(sizeof(struct Node));
    if (!head) {
        printf("Memory allocation failed\n");
        return NULL;
    }

    head->data = 0;  
    head->next = NULL;
    
    struct Node *current = head;
    for (int i = 1; i < size; i++) {
        current->next = malloc(sizeof(struct Node));
        if (!current->next) {
            printf("Memory allocation failed\n");
            return head; 
        }
        current = current->next;
        current->data = i; 
        current->next = NULL;
    }
    
    return head;
}

void freeList(struct Node *head) {
    struct Node *current = head;
    while (current) {
        struct Node *temp = current;
        current = current->next;
        free(temp);
	temp = NULL;
    }
}

void printList(struct Node *head) {
	struct Node *iter = (struct Node*)malloc(sizeof(struct Node));
	iter = head;

	while(iter) {
		printf("%d\n", iter->data);
		iter = iter->next;
	}
}


void incOmp(struct Node *head) {
	struct Node *current = (struct Node*)malloc(sizeof(struct Node));
	if(!current) {
		printf("Memory not allocated\n");
		return;
	}

	current = head;
	printf("In OMP\n");
	#pragma omp parallel
	{
		while(1) {
			#pragma omp critical
			{
				if(current) {
					current->data++;
					current = current->next;
				}
			}
			if(!current) {
				break;
			}
		}
	}
}

void incOmpTask(struct Node *head) {
	struct Node *current = (struct Node*)malloc(sizeof(struct Node));
	if(!current) {
		printf("Memory not allocated\n");
		return;
	}
	current = head;
	omp_set_num_threads(10);
	#pragma omp parallel
	{
		#pragma omp single
		{
			while(current) {
				printf("SINGLE REGION thread number %d\n", omp_get_thread_num());
				#pragma omp task firstprivate(current)
				{
				printf("executed by %d\n", omp_get_thread_num());
				//usleep(1000000);
				current->data++;
				}
				current = current->next;
			}
		}
	}
}

int main() {
	int size = 100;
	struct Node *head = createList(size);
	
	printList(head);

	incOmp(head);

	printList(head);

	incOmpTask(head);

	printList(head);

}

	
