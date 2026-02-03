import os
import pandas as pd
from pathlib import Path
from collections import defaultdict


def mm4_doc_to_visual(doc):
    """
    Get the image from the document
    """
    return [doc["image"].convert("RGB")]


def mm4_doc_to_text(doc):
    """
    Get the question text
    """
    question = doc["question"]
    return question


def mm4_doc_to_target(doc):
    """
    Get the correct answer
    """
    return doc["answer"]


def mm4_doc_to_choice(doc):
    """
    Extract choices from the question text
    """
    question = doc["question"]

    # Parse options from question
    # Format: "Question text Options: A: choice1, B: choice2, C: choice3, D: choice4"
    if "Options:" in question:
        options_part = question.split("Options:")[-1].strip()
        # Split by comma and then by colon to get choices
        choices = []
        for option in options_part.split(","):
            option = option.strip()
            if ":" in option:
                choice = option.split(":", 1)[1].strip()
                choices.append(choice)
        return choices

    # If no options found, return empty list
    return []


def mm4_aggregate_by_image(results):
    """
    Custom aggregation function for MM4 that groups by image
    and calculates the average number of correct answers per image

    Args:
        results: List of result dictionaries from process_results
    """
    # Group results by figure_name and count correct answers
    image_groups = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        figure_name = result.get('figure_name', 'unknown')
        correct = result.get('correct', 0)

        image_groups[figure_name]["correct"] += correct
        image_groups[figure_name]["total"] += 1

    # Count images by number of correct answers
    count_ge_1 = 0  # >= 1 correct
    count_ge_2 = 0  # >= 2 correct
    count_ge_3 = 0  # >= 3 correct
    count_ge_4 = 0  # >= 4 correct

    print(f"\n[MM4] Per-image statistics:")
    for figure_name, stats in sorted(image_groups.items()):
        correct = stats["correct"]
        total = stats["total"]

        if correct >= 1:
            count_ge_1 += 1
        if correct >= 2:
            count_ge_2 += 1
        if correct >= 3:
            count_ge_3 += 1
        if correct >= 4:
            count_ge_4 += 1

    total_images = len(image_groups)

    print(f"\n[MM4] Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Images with >= 1 correct: {count_ge_1} ({count_ge_1/total_images*100:.1f}%)")
    print(f"  Images with >= 2 correct: {count_ge_2} ({count_ge_2/total_images*100:.1f}%)")
    print(f"  Images with >= 3 correct: {count_ge_3} ({count_ge_3/total_images*100:.1f}%)")
    print(f"  Images with >= 4 correct: {count_ge_4} ({count_ge_4/total_images*100:.1f}%)\n")

    # Return average (for compatibility)
    correct_counts = [stats["correct"] for stats in image_groups.values()]
    mean_correct = sum(correct_counts) / len(correct_counts) if correct_counts else 0
    return mean_correct


def mm4_aggregate_ge_1(results):
    """Count images with >= 1 correct answers"""
    image_groups = defaultdict(lambda: {"correct": 0})
    for result in results:
        figure_name = result.get('figure_name', 'unknown')
        correct = result.get('correct', 0)
        image_groups[figure_name]["correct"] += correct

    count = sum(1 for stats in image_groups.values() if stats["correct"] >= 1)
    return count


def mm4_aggregate_ge_2(results):
    """Count images with >= 2 correct answers"""
    image_groups = defaultdict(lambda: {"correct": 0})
    for result in results:
        figure_name = result.get('figure_name', 'unknown')
        correct = result.get('correct', 0)
        image_groups[figure_name]["correct"] += correct

    count = sum(1 for stats in image_groups.values() if stats["correct"] >= 2)
    return count


def mm4_aggregate_ge_3(results):
    """Count images with >= 3 correct answers"""
    image_groups = defaultdict(lambda: {"correct": 0})
    for result in results:
        figure_name = result.get('figure_name', 'unknown')
        correct = result.get('correct', 0)
        image_groups[figure_name]["correct"] += correct

    count = sum(1 for stats in image_groups.values() if stats["correct"] >= 3)
    return count


def mm4_aggregate_ge_4(results):
    """Count images with >= 4 correct answers"""
    image_groups = defaultdict(lambda: {"correct": 0})
    for result in results:
        figure_name = result.get('figure_name', 'unknown')
        correct = result.get('correct', 0)
        image_groups[figure_name]["correct"] += correct

    count = sum(1 for stats in image_groups.values() if stats["correct"] >= 4)
    return count


def mm4_process_results(doc, results):
    """
    Process results and add figure_name for aggregation
    """
    pred = results[0] if isinstance(results, list) and len(results) > 0 else results
    target = mm4_doc_to_target(doc)

    # Compare prediction with target
    correct = 1 if str(pred).strip().upper() == str(target).strip().upper() else 0

    # Return a dict with all info we need
    result_dict = {
        "correct": correct,
        "figure_name": doc.get("figure_name", ""),
        "category": doc.get("category", ""),
    }

    # Return multiple metrics (all with the same data)
    return {
        "mm4_avg": result_dict,
        "mm4_ge_1": result_dict,
        "mm4_ge_2": result_dict,
        "mm4_ge_3": result_dict,
        "mm4_ge_4": result_dict,
    }
