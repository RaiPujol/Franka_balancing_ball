"""Render five matching demonstrations for a selected offline candidate."""
import argparse,json
from pathlib import Path
from render_methods import controller_factory,LABELS,show_exact_stop_reason
from wrench_shapes import run as shape_run
from wrench_center_demos import WrenchVideo,run_one


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=list(LABELS),required=True)
    p.add_argument('--output-dir',default='output/selected_videos')
    a=p.parse_args(); root=Path(a.output_dir)/a.method;root.mkdir(parents=True,exist_ok=True)
    name=LABELS[a.method];factory=controller_factory(a.method); summaries={}
    for task in ('circle','rounded_square','figure_eight','centering','carrying'):
        path=root/f'{task}.mp4';print('Rendering',name,task,flush=True)
        if task in ('circle','rounded_square','figure_eight'):
            r=shape_run(task,config={'ball_initial_offset_m':[.018,-.012]},
                period_s=16. if task=='rounded_square' else 8.,
                output_dir=root/task,video=str(path),sensor_model='joint_torque',
                joint_torque_std=.0005,joint_velocity_std=.0001,delay_s=.05,seed=401,
                controller_factory=factory,controller_label=name)
            if r['stop_reason']:show_exact_stop_reason(path,r['stop_reason'])
            rms=r['plateau_tracking_rms_m']
            passed=bool(r['numerical_pass'] and rms is not None and rms<.005 and r['max_tray_position_error_m']<.02)
        else:
            video=WrenchVideo(path,0.,0.,50.,.0001,'joint_torque',.0005)
            try:
                r=run_one((.018,-.012),task,video=video,label=f'{name}: {task}',
                    sensor_model='joint_torque',joint_torque_std=.0005,joint_velocity_std=.0001,
                    delay_s=.05,seed=401,controller_factory=factory,output_csv=root/f'{task}.csv')
            finally:video.close()
            passed=r['success']
        summaries[task]={'performance_pass':passed,'video':path.name,'results':r}
        (root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(task, 'PASS' if passed else 'FAIL',flush=True)


if __name__=='__main__':main()
