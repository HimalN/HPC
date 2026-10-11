/*
 * Student Name: Himal Perera
 * Module      : 6CS005 - High Performance Computing
 * Task        : Task 1 - Word Occurrence Counting using Multithreading (Pthreads)
 *
 * COMPILE: gcc -std=c11 -O2 -Wall -Wextra -pthread word_count.c -o word_count
 * RUN    : ./word_count WordOccurrenceDataset.txt 4
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define BUCKETS 8192
#define CHUNK_SIZE 128

// Structure for the hash table nodes
typedef struct WordNode {
    const char *word;
    size_t count;
    struct WordNode *next;
} WordNode;

// Shared data structure for threads
typedef struct {
    char **words;
    size_t total;
    size_t next_index;
    size_t unique_count;
    WordNode *table[BUCKETS];
    pthread_mutex_t work_mutex;
    pthread_mutex_t table_mutex;
} SharedData;

// Simple hash function (djb2)
unsigned long hash_word(const char *word) {
    unsigned long hash = 5381;
    while (*word) {
        hash = hash * 33 + (unsigned char)*word++;
    }
    return hash % BUCKETS;
}

// Thread worker function
void *count_words(void *arg) {
    SharedData *data = (SharedData *)arg;

    while (1) {
        // --- Dynamic Work Allocation ---
        pthread_mutex_lock(&data->work_mutex);
        size_t start = data->next_index;
        size_t end = start + CHUNK_SIZE;
        if (end > data->total) end = data->total;
        data->next_index = end;
        pthread_mutex_unlock(&data->work_mutex);

        if (start >= data->total) break;

        // --- Process Chunk ---
        for (size_t i = start; i < end; i++) {
            const char *word = data->words[i];
            unsigned long bucket = hash_word(word);

            // --- Synchronize Shared Hash Table Updates ---
            pthread_mutex_lock(&data->table_mutex);

            WordNode *node = data->table[bucket];
            while (node && strcmp(node->word, word) != 0) {
                node = node->next;
            }

            if (node) {
                node->count++;
            } else {
                node = malloc(sizeof(WordNode));
                if (!node) {
                    fprintf(stderr, "Memory allocation failed\n");
                    exit(EXIT_FAILURE);
                }

                node->word = word;
                node->count = 1;
                node->next = data->table[bucket];
                data->table[bucket] = node;
                data->unique_count++;
            }

            pthread_mutex_unlock(&data->table_mutex);
        }
    }

    return NULL;
}

// Comparator function for sorting alphabetically
int compare_words(const void *a, const void *b) {
    const WordNode *x = *(const WordNode *const *)a;
    const WordNode *y = *(const WordNode *const *)b;
    return strcmp(x->word, y->word);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <filename> <threads>\n", argv[0]);
        return 1;
    }

    // 1. Parse Thread Count
    char *endptr;
    long requested = strtol(argv[2], &endptr, 10);

    if (*endptr != '\0' || requested < 1 || requested > 1024) {
        fprintf(stderr, "Thread count must be between 1 and 1024\n");
        return 1;
    }
    int num_threads = (int)requested;

    // 2. Read File Data
    FILE *input = fopen(argv[1], "r");
    if (!input) {
        perror("Cannot open input file");
        return 1;
    }

    size_t capacity = 4096;
    size_t total = 0;
    char **words = malloc(capacity * sizeof(char *));
    if (!words) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(input);
        return 1;
    }

    char buffer[256];
    while (fscanf(input, "%255s", buffer) == 1) {
        if (total == capacity) {
            capacity *= 2;
            char **temp = realloc(words, capacity * sizeof(char *));
            if (!temp) {
                fprintf(stderr, "Memory allocation failed\n");
                return 1;
            }
            words = temp;
        }

        words[total] = malloc(strlen(buffer) + 1);
        if (!words[total]) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        strcpy(words[total], buffer);
        total++;
    }

    if (ferror(input)) {
        fprintf(stderr, "Error reading input file\n");
        fclose(input);
        return 1;
    }
    fclose(input);

    // 3. Initialize Shared Data
    SharedData *data = calloc(1, sizeof(SharedData));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    data->words = words;
    data->total = total;

    pthread_mutex_init(&data->work_mutex, NULL);
    pthread_mutex_init(&data->table_mutex, NULL);

    // 4. Create and Run Threads
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    if (!threads) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    int started = 0;
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, count_words, data) != 0) {
            fprintf(stderr, "Failed to create thread %d\n", i);
            break;
        }
        started++;
    }

    if (started == 0) return 1;

    for (int i = 0; i < started; i++) {
        pthread_join(threads[i], NULL);
    }

    // 5. Sort the Results
    WordNode **sorted = malloc((data->unique_count ? data->unique_count : 1) * sizeof(WordNode *));
    if (!sorted) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    size_t index = 0;
    for (int i = 0; i < BUCKETS; i++) {
        WordNode *node = data->table[i];
        while (node) {
            sorted[index++] = node;
            node = node->next;
        }
    }

    qsort(sorted, index, sizeof(WordNode *), compare_words);

    // 6. Write Results to Output File
    FILE *output = fopen("result.txt", "w");
    if (!output) {
        perror("Cannot create result.txt");
        return 1;
    }

    fprintf(output, "Word\tFrequency\n");
    for (size_t i = 0; i < index; i++) {
        fprintf(output, "%s\t%zu\n", sorted[i]->word, sorted[i]->count);
    }
    fclose(output);

    // 7. Print Summary and Clean Up Memory
    printf("Threads requested: %d\n", num_threads);
    printf("Threads created: %d\n", started);
    printf("Total words: %zu\n", total);
    printf("Unique words: %zu\n", index);
    printf("Output saved to result.txt\n");

    for (int i = 0; i < BUCKETS; i++) {
        WordNode *node = data->table[i];
        while (node) {
            WordNode *next = node->next;
            free(node);
            node = next;
        }
    }

    for (size_t i = 0; i < total; i++) free(words[i]);
    free(words);
    free(sorted);
    free(threads);
    pthread_mutex_destroy(&data->work_mutex);
    pthread_mutex_destroy(&data->table_mutex);
    free(data);

    return 0;
}
