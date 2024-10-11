#-----------------------------------------------------------------------------------------------
!/bin/bash

source_values=('schoolA,schoolB,schoolC' 'schoolA,schoolB,schoolD' 'schoolA,schoolC,schoolD' 'schoolB,schoolC,schoolD')
target_values=('schoolD' 'schoolC' 'schoolB' 'schoolA')
model_file_values=('ABC_D.pth' 'ABD_C.pth' 'ACD_B.pth' 'BCD_A')
folder_values=('../data/A+B+C_D' '../data/A+B+D_C' '../data/A+C+D_B' '../data/B+C+D_A')

script_names=('scripts/main_irt_cross_school.py' 'scripts/main_mirt_cross_school.py' 'scripts/main_ncdm_cross_school.py' 'scripts/main_kscd_cross_school.py')
prefix_values=('source_model/cross_school/irt/' 'source_model/cross_school/mirt/' 'source_model/cross_school/ncdm/' 'source_model/cross_school/kscd/')

batch_size_val=256

if_source_train_values=(1 0 0)
if_target_migration_values=(0 1 2)

for ((k=0; k<${#script_names[@]}; k++))
do
    script_name=${script_names[$k]}
    prefix_val=${prefix_values[$k]}

    for ((i=0; i<${#source_values[@]}; i++))
    do
        source_val=${source_values[$i]}
        target_val=${target_values[$i]}
        model_file_val=${prefix_val}${model_file_values[$i]}
        folder_val=${folder_values[$i]}

        echo "Setting parameters: source=$source_val, target=$target_val, model_file=$model_file_val, folder=$folder_val"

        for ((j=0; j<${#if_source_train_values[@]}; j++))
        do
            if_source_train_val=${if_source_train_values[$j]}
            if_target_migration_val=${if_target_migration_values[$j]}
            echo "Running $script_name with if_source_train=$if_source_train_val and if_target_migration=$if_target_migration_val"
            python $script_name --source $source_val --target $target_val --model_file $model_file_val --folder $folder_val --if_source_train $if_source_train_val --if_target_migration $if_target_migration_val --batch_size $batch_size_val
        done
    done
done
