#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# DONE: 0. Fill in your information in the programming header below
# PROGRAMMER: Demondre Livingston
# DATE CREATED:
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir

# Imports classifier function for using CNN to classify images
from classifier import classifier


# Main program function defined below
def main():
    # DONE: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # DONE: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()

    # Accesses values of Arguments 1 and 2 by printing them
    print("Command Line Arguments:\n  dir=", in_arg.dir,
          "\n arch =", in_arg.arch, "\n dogfile =", in_arg.dogfile)

    # DONE: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # Temporary code to print 10 key-value pairs & makes sure there are 40 pairs,
    # one for each file in pet_images/

    # print("\nanswers_dic has", len(answers_dic),
    #       "key-value pairs.\nBelow are 10 of them:")
    # prnt = 0
    # for key in answers_dic:
    #     if prnt < 10:
    #         print("%2d key: %-30s label: %-26s" % (prnt + 1, key, answers_dic[key]))
    #     prnt += 1

    # DONE: 4. Define classify_images() function to create the classifier
    # labels wit h the classifier function uisng in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # DONE: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)
    # print("\n MATCH:")
    # n_match = 0
    # n_notmatch = 0
    # for key in result_dic:
    #     if result_dic[key][2] == 1:
    #         n_match += 1
    #         # print("Real: %-26s Classifier: %-30s PetLabelDog: %1d ClassLabelDog: %1d" %
    #         #       (result_dic[key][0], result_dic[key][1], result_dic[key][3], result_dic[key][4]))

    # print("\n NOT A MATCH:")
    # for key in result_dic:
    #     if key in result_dic:
    #         if result_dic[key][2] == 0:
    #             n_notmatch += 1
    #             # print("Real: %-26s Classifier: %-30s PetLabelDog: %1d ClassLabelDog: %1d" %
    #             #       (result_dic[key][0], result_dic[key][1], result_dic[key][3], result_dic[key][4]))

    # print("\n#Total Images", n_match + n_notmatch, "#Matches:", n_match,
    #       "# Not Matches:", n_notmatch)

    # DONE: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # #Temporary code for checking results
    # #checks calculations
    # n_images=len(result_dic)
    # n_pet_dog=0
    # n_class_cdog=0
    # n_class_cnotd=0
    # n_match_breed=0
    # for key in result_dic:
    #     #match (if dog then breed match)
    #     if result_dic[key][2]==1:
    #         #isa dog (pet label)&breed match
    #         if result_dic[key][3]==1:
    #             #isa dog (pet label)&breed match
    #             n_pet_dog+=1
    #             #isa dog(classifier label)&breed match
    #             if result_dic[key][4]==1:
    #                 n_class_cdog+=1
    #                 n_match_breed+=1
    #         #NOT dog(pet_label)
    #         else:
    #             #NOT dog (Classifier label)
    #             if result_dic[key][4]==0:
    #                 n_class_cnotd+=1

    #     #not = match(not a breed match if a dog)
    #     else:
    #         #not=match
    #         #isa dog(pet_label)
    #         if result_dic[key][3]==1:
    #             n_pet_dog+=1
    #             #isa dog(Classifier label)
    #             if result_dic[key][4]==1:
    #                 n_class_cdog+=1
    #         else:
    #             #NOT dog(classifier label)
    #             if result_dic[key][4]==0:
    #                 n_class_cnotd+=1

    # #calculates rest of statistics
    # n_pet_notd=n_images-n_pet_dog
    # pct_corr_dog=(n_class_cdog/n_pet_dog)*100
    # pct_corr_notdog=(n_class_cnotd/n_pet_notd)*100
    # pct_corr_breed=(n_match_breed/n_pet_dog)*100

    # #prints calculated run stats
    # print("\n **Function's Stats:")
    # print("N Images: %2d N Dog Images: %2d N NotDog Images:%2d\nPet Corr dog: %5.1f Pet Corr NOTdog: %5.1f Pet Corr Breed: %5.1f"
    #     %(results_stats_dic['n_images'],results_stats_dic['n_dogs_img'],results_stats_dic['n_notdogs_img'],
    #     results_stats_dic['pet_correct_dogs'],results_stats_dic['n_correct_notdogs'],results_stats_dic['pet_correct_breed']))
    # print("\n **Check Stats:")
    # print("N Images: %2d N Dog Images: %2d N NotDog Images: %2d \nPct Corr Dog: %5.1f Pct Corr NOTdog: %5.1f Pet Corr Breed: %5.1f"
    # %(n_images,n_pet_dog,n_pet_notd,pct_corr_dog,pct_corr_notdog,pct_corr_breed))

    # DONE: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # DONE: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # DONE: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\nTotal Elapsed Runtime:", str(int((tot_time / 3600))) + ":" +
          str(int(((tot_time % 3600) / 60))) + ":" +
          str(int(((tot_time % 3600) % 60))))


# TODO: 2.-to-7. Define all the function below. Notice that the input
# paramaters and return values have been left in the function's docstrings.
# This is to provide guidance for acheiving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the folder my_folder')
    parser.add_argument('--arch', type=str, default='vgg',
                        help='chosen model')
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='text file that has dognames')
    in_args = parser.parse_args()

    return in_args


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    # Retrieve the filenames from folder pet_images/
    in_files = listdir(image_dir)

    # Creates empty dictionary named petlabels_dic
    petlabels_dic = dict()

    for index in range(0, len(in_files), 1):

        # skips file if starts with . (like .DS_Store of Mac OSX) because it
        # isn't an pet image file
        if in_files[index][0] != ".":

            # Uses split to extract words of filename into list image_name
            image_name = in_files[index].split("_")

            # creates temporary label variable to hold pet label name extracted
            pet_label = ""

            # Processes each of the character strings(words split by '_' in
            # list image_name by processing each word - only adding to pet_label
            # if word is all letters - then process by putting blanks between
            # these words and putting them in all lowercase letters
            for word in image_name:
                # Only add to pet_label if word is all letters add blank at end
                if word.isalpha():
                    pet_label += word.lower() + " "
            # strips off trailing whitespace
            pet_label = pet_label.strip()
            # if filename doesn't already exist in dictionary add it and it's
            # pet label - otherwise print an error message because indicates
            # duplicate files(filenames)
            if in_files[index] not in petlabels_dic:
                petlabels_dic[in_files[index]] = pet_label
            else:
                print("Warning: Duplicate files exist in directory",
                      in_files[index])

    # returns dictionary of labels
    return petlabels_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    # Creates dictionary that will move all the results key=filename
    # value =list[pet_label,Classifier Label, Match(1=yes,0=no)]
    results_dic = dict()

    # Process all files in the petlabels_dic- use images_dir to give fullpath
    for key in petlabel_dic:

        # Runs classifier function to classigy the images classifier function
        # inputs: path + filename and model, returns model_label
        # as classifier label
        model_label = classifier(images_dir + key, model)

        # Processes the results so they can be compared with pet image labels
        # set labels to lowercase(lower) and stripping off whitespace(strip)
        model_label = model_label.lower()
        model_label = model_label.strip()

        # defines truth as pet image label and trys to find it using find()
        # string function to find it within classifier label(model_label).
        truth = petlabel_dic[key]
        found = model_label.find(truth)

        # if found (o or greater) then make sure true answer wasn't found within
        # another word and thus not really found, if truely found then add to
        # results dictionary and set match=1(yes)otherwise as match=0(no)
        if found >= 0:
            if ((found == 0 and len(truth) == len(model_label)) or
                ((found == 0) or model_label[found - 1] == " ") and
                        ((found + len(truth) == len(model_label)) or
                         (model_label[found + len(truth):found + len(truth) + 1] in
                          (",", " "))
                         )
                ):
                # found label as stand-alone term(not within label)
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 1]
            # found within a word/term not a label existing on its own
            else:
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 0]

        # if not found set results dictionary with match=0(no)
        else:
            if key not in results_dic:
                results_dic[key] = [truth, model_label, 0]

    # Return results dictionary
    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dognames_dic = dict()
    dogfile = open(dogsfile, 'r')
    for dog in dogfile:
        if dog not in dognames_dic:
            dogs = dog.rstrip()
            dognames_dic[dogs] = [1]
        else:
            print("Warning: Duplicate files exist in directory",
                  dog)

    for key in results_dic:

        # Pet Image Label is of Dog(e.g. found in dognames_dic)
        if results_dic[key][0]in dognames_dic:

            # Classifier Label Is image of Dog(e.g. found in dognames_dic)
            # appends(1,1)because both labels are dogs
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1, 1))

            # Classifier Label Is image of Dog(e.g. found in dognames_dic)
            # appends(1,0)because only pet label is a dog
            else:
                results_dic[key].extend((1, 0))

        # Pet Image IS NOT a Dog image(e.d. NOT found in dognames_dic)
        else:
            # classifier Label IS image of Dog(e.g. found in dognames_dic)
            # appends(0,1)becuase only Classifier label is a dog
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((0, 1))

            # classifier Label IS NOT image of Dog(e.g. found in dognames_dic)
            # appends(0,1)because both labels aren't dogs
            else:
                results_dic[key].extend((0, 0))


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    # creates empty dictionary for results_stats
    results_stats = dict()

    # Set all counters to initial values of zero so that they can
    # be incremented while processing through the images in results_dic
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0

    # process through the results dictionary
    for key in results_dic:
        # Labels match Exactly
        if results_dic[key][2] == 1:
            results_stats['n_match'] += 1

        # put Image Label is a Dog AND Labels match- counts Correct Breed
        if sum(results_dic[key][2:]) == 3:
            results_stats['n_correct_breed'] += 1

        # Put Image Label is a Dog counts number of dogs images
        if results_dic[key][3] == 1:
            results_stats['n_dogs_img'] += 1

            # Classiger classifies image as Dog[& pet image is a dog]
            # counts number of correct dog classifications
            if results_dic[key][4] == 1:
                results_stats['n_correct_dogs'] += 1

        # Pet Image Label is NOT a Dog
        else:
            # Classifier classifies images as NOT a Dog  &pet image isn't a dog
            # counts number of correct NOT dog clasifications
            if results_dic[key][4] == 0:
                results_stats['n_correct_notdogs']+= 1

    # Calculate number of total images
    results_stats['n_images'] = len(results_dic)

    # calculate number of not-a-dog images using - images & dog images counts
    results_stats['n_notdogs_img'] = (
        results_stats['n_images']-results_stats['n_dogs_img'])

    # calculate % of correct for matches
    results_stats['pet_match'] = (
        results_stats['n_match']/results_stats['n_images'])*100.0

    # Calculates % correct dogs
    results_stats['pet_correct_dogs'] = (
        results_stats['n_correct_dogs']/results_stats['n_dogs_img'])*100.0

    # Calculate % correct breed of dog
    results_stats['pet_correct_breed'] = (
        results_stats['n_correct_breed']/results_stats['n_dogs_img'])*100.0

    # Calculate % correct not-a-dog images
    # uses condtional statement for when no 'not a dog' images were submitted
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pet_correct_notdogs'] = (
            results_stats['n_correct_notdogs']/results_stats['n_notdogs_img'])*100.0
    else:
        results_stats['pet_correct_notdogs'] = 0.0

    # return results_stats dictionary
    return results_stats


def print_results(results_dic, results_stats, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    # Prints Summary statistics over the run
    print("*** Results Summary for CNN Model Architecture", model.upper(), "***")
    print("%20s: %3d" % ('N Images', results_stats['n_images']))
    print("%20s: %3d" % ('N Dog Images', results_stats['n_dogs_img']))
    print("%20s: %3d" % ('N Not-Dog Images', results_stats['n_notdogs_img']))

    # Prints summary statistics (percentages) on Model Run
    print(" ")
    for key in results_stats:
        if key[0] == "p":
            print("%20s: %5.1f" % (key, results_stats[key]))

    # IF print_incorrect_dogs == True AND there were images incorrectly
    # classified as dogs or vice versa-print out these cases
    if(print_incorrect_dogs and ((results_stats['n_correct_dogs']+results_stats['n_correct_notdogs'])
                                 != results_stats['n_images'])
       ):
        print("\nINCORRECT Dog/Not Dog Assignments:")

        # process through results dict, printing incorrectly classified dogs
        for key in results_dic:

            # Pet Image Label is a Dog-Classified as NOT-A-DOG -OR-
            # Pet Image Label is NOT-A-DOG - Classified  as a-DOG
            if sum(results_dic[key][3:]) == 1:
                print("Real: %-26s Classifier: %-30s" %
                      (results_dic[key][0], results_dic[key][1]))

    # IF print_incorrect_breed==True AND there were dogs whose breeds
    # were incorrectly classified - print out these cases
    if(print_incorrect_breed and (results_stats['n_correct_dogs'] != results_stats['n_correct_breed'])
       ):
        print("\nINCORRECT Dog Breed Assignment:")

        # Process through results dict, printing incorrectly classified breeds
        for key in results_dic:

            # Pet Image Label is-a-dog, classified as-a-dog but is WRONG breed
            if(sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0):
                print("Real: %-26s Classifier: %-30s" %
                      (results_dic[key][0], results_dic[key][1]))



        # Call to main function to run the program
if __name__ == "__main__":
    main()
