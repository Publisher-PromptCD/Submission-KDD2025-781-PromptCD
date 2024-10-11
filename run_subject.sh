#------------------------------------------------------------------------------------------------------------
#!/bin/bash

source_values=('bio,mat' 'bio,phy' 'mat,phy')
target_values=('phy' 'mat' 'bio' )
model_file_values=('bm_p.pth' 'bp_m.pth' 'mp_b.pth')
folder_values=('../data/science_2+1/b_m+p' '../data/science_2+1/b_p+m' '../data/science_2+1/m_p+b')

script_names=('scripts/main_irt_cross_subject.py' 'scripts/main_mirt_cross_subject.py' 'scripts/main_ncdm_cross_subject.py' 'scripts/main_kscd_cross_subject.py')
prefix_values=('source_model/cross_subject/irt/' 'source_model/cross_subject/mirt/' 'source_model/cross_subject/ncdm/' 'source_model/cross_subject/kscd/')

if_source_train_values=(1 0 0)
if_target_migration_values=(0 1 2)

batch_size_val=256

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

#-------------------------------------------------------------------------------------------------------
source_values=('chi,geo' 'chi,his' 'his,geo')
target_values=('his' 'geo' 'chi' )
model_file_values=('cg_h.pth' 'ch_g.pth' 'hg_c.pth')
folder_values=('../data/humanities_2+1/c+g_h' '../data/humanities_2+1/c+h_g' '../data/humanities_2+1/h+g_c')

script_names=('scripts/main_irt_cross_subject.py' 'scripts/main_mirt_cross_subject.py' 'scripts/main_ncdm_cross_subject.py' 'scripts/main_kscd_cross_subject.py')
prefix_values=('source_model/cross_subject/irt/' 'source_model/cross_subject/mirt/' 'source_model/cross_subject/ncdm/' 'source_model/cross_subject/kscd/')

if_source_train_values=(1 0 0)
if_target_migration_values=(0 1 2)

batch_size_val=256

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