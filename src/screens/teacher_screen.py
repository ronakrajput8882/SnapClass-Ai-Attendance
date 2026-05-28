import streamlit as st
import numpy as np
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo

from src.ui.base_layout import (
    style_background_dashboard,
    style_base_layout,
)
from src.screens.teacher_screen_login import (
    teacher_screen_login,
)

from src.screens.teacher_screen_register import (
    teacher_screen_register,
)
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.components.subject_card import subject_card

from src.database.db import (
    check_teacher_exists,
    create_teacher,
    teacher_login,
    get_teacher_subjects,
    get_attendance_for_teacher,
)

from src.database.config import supabase

from src.components.dialog_create_subject import create_subject_dialog
from src.components.dialog_share_subject import share_subject_dialog
from src.components.dialog_add_photo import add_photos_dialog
from src.components.dialog_attendance_results import (
    attendance_result_dialog,
)
from src.components.dialog_voice_attendance import (
    voice_attendance_dialog,
)

from src.pipelines.face_pipeline import predict_attendance


def teacher_screen():
    style_background_dashboard()
    style_base_layout()

    if "teacher_data" in st.session_state:
        teacher_dashboard()

    elif (
        "teacher_login_type" not in st.session_state
        or st.session_state.teacher_login_type == "login"
    ):
        teacher_screen_login()

    elif st.session_state.teacher_login_type == "register":
        teacher_screen_register()


def teacher_dashboard():
    teacher_data = st.session_state.teacher_data

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge",
    )

    with c1:
        header_dashboard()

    with c2:
        st.subheader(
            f"Welcome, {teacher_data['name']}"
        )

        if st.button(
            "Logout",
            type="secondary",
            key="loginbackbtn",
            shortcut="control+backspace",
        ):
            st.session_state["is_logged_in"] = False
            del st.session_state.teacher_data
            st.rerun()

    st.space()

    if "current_teacher_tab" not in st.session_state:
        st.session_state.current_teacher_tab = (
            "take_attendance"
        )

    tab1, tab2, tab3 = st.columns(3)

    with tab1:
        type1 = (
            "primary"
            if st.session_state.current_teacher_tab
            == "take_attendance"
            else "tertiary"
        )

        if st.button(
            "Take Attendance",
            type=type1,
            width="stretch",
            icon=":material/ar_on_you:",
        ):
            st.session_state.current_teacher_tab = (
                "take_attendance"
            )
            st.rerun()

    with tab2:
        type2 = (
            "primary"
            if st.session_state.current_teacher_tab
            == "manage_subjects"
            else "tertiary"
        )

        if st.button(
            "Manage Subjects",
            type=type2,
            width="stretch",
            icon=":material/book_ribbon:",
        ):
            st.session_state.current_teacher_tab = (
                "manage_subjects"
            )
            st.rerun()

    with tab3:
        type3 = (
            "primary"
            if st.session_state.current_teacher_tab
            == "attendance_records"
            else "tertiary"
        )

        if st.button(
            "Attendance Records",
            type=type3,
            width="stretch",
            icon=":material/cards_stack:",
        ):
            st.session_state.current_teacher_tab = (
                "attendance_records"
            )
            st.rerun()

    st.divider()

    if (
        st.session_state.current_teacher_tab
        == "take_attendance"
    ):
        teacher_tab_take_attendance()

    if (
        st.session_state.current_teacher_tab
        == "manage_subjects"
    ):
        teacher_tab_manage_subjects()

    if (
        st.session_state.current_teacher_tab
        == "attendance_records"
    ):
        teacher_tab_attendance_records()

    footer_dashboard()


def teacher_tab_take_attendance():
    teacher_id = st.session_state.teacher_data[
        "teacher_id"
    ]

    st.header("Take AI Attendance")

    if "attendance_images" not in st.session_state:
        st.session_state.attendance_images = []

    subjects = get_teacher_subjects(teacher_id)

    if not subjects:
        st.warning(
            "You havent created any subjects yet!"
        )
        return

    subject_options = {
        f"{s['name']} - {s['subject_code']}":
        s["subject_id"]
        for s in subjects
    }

    col1, col2 = st.columns(
        [3, 1],
        vertical_alignment="bottom",
    )

    with col1:
        selected_subject_label = st.selectbox(
            "Select Subject",
            options=list(subject_options.keys()),
        )

    with col2:
        if st.button(
            "Add Photos",
            type="primary",
            width="stretch",
        ):
            add_photos_dialog()

    selected_subject_id = subject_options[
        selected_subject_label
    ]

    st.divider()

    if st.session_state.attendance_images:
        st.header("Added Photos")

        gallery_cols = st.columns(4)

        for idx, img in enumerate(
            st.session_state.attendance_images
        ):
            with gallery_cols[idx % 4]:
                st.image(
                    img,
                    width="stretch",
                    caption=f"Photo {idx+1}",
                )

    has_photos = bool(
        st.session_state.attendance_images
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(
            "Clear all photos",
            disabled=not has_photos,
        ):
            st.session_state.attendance_images = []
            st.rerun()

    with c2:
        if st.button(
            "Run Face Analysis",
            disabled=not has_photos,
        ):
            with st.spinner(
                "Deep scanning classroom photos..."
            ):
                all_detected_ids = {}

                for idx, img in enumerate(
                    st.session_state
                    .attendance_images
                ):
                    img_np = np.array(
                        img.convert("RGB")
                    )

                    detected, _, _ = (
                        predict_attendance(img_np)
                    )

                    if detected:
                        for sid in detected:
                            student_id = int(sid)

                            all_detected_ids.setdefault(
                                student_id,
                                [],
                            ).append(
                                f"Photo {idx+1}"
                            )

                enrolled_res = (
                    supabase
                    .table("subject_students")
                    .select(
                        "*, students(*)"
                    )
                    .eq(
                        "subject_id",
                        selected_subject_id,
                    )
                    .execute()
                )

                enrolled_students = (
                    enrolled_res.data
                )

                if not enrolled_students:
                    st.warning(
                        "No students enrolled"
                    )

                else:
                    results = []
                    attendance_to_log = []

                    current_timestamp = (
                        datetime.now()
                        .strftime(
                            "%Y-%m-%dT%H:%M:%S"
                        )
                    )

                    for node in (
                        enrolled_students
                    ):
                        student = node[
                            "students"
                        ]

                        sources = (
                            all_detected_ids.get(
                                int(
                                    student[
                                        "student_id"
                                    ]
                                ),
                                [],
                            )
                        )

                        is_present = (
                            len(sources) > 0
                        )

                        results.append(
                            {
                                "Name":
                                student["name"],
                                "ID":
                                student[
                                    "student_id"
                                ],
                                "Source":
                                ", ".join(
                                    sources
                                )
                                if is_present
                                else "-",
                                "Status":
                                "✅ Present"
                                if is_present
                                else "❌ Absent",
                            }
                        )

                        attendance_to_log.append(
                            {
                                "student_id":
                                student[
                                    "student_id"
                                ],
                                "subject_id":
                                selected_subject_id,
                                "timestamp":
                                current_timestamp,
                                "is_present":
                                bool(
                                    is_present
                                ),
                            }
                        )

                attendance_result_dialog(
                    pd.DataFrame(results),
                    attendance_to_log,
                )

    with c3:
        if st.button(
            "Use Voice Attendance",
            type="primary",
        ):
            voice_attendance_dialog(
                selected_subject_id
            )


def teacher_tab_manage_subjects():
    teacher_id = st.session_state.teacher_data[
        "teacher_id"
    ]

    st.header("Manage Subjects")

    if st.button("Create New Subject"):
        create_subject_dialog(teacher_id)

    subjects = get_teacher_subjects(
        teacher_id
    )

    if not subjects:
        st.info("NO SUBJECTS FOUND")
        return

    for sub in subjects:
        stats = [
            (
                "🫂",
                "Students",
                sub["total_students"],
            ),
            (
                "🕰️",
                "Classes",
                sub["total_classes"],
            ),
        ]

        subject_card(
            name=sub["name"],
            code=sub["subject_code"],
            section=sub["section"],
            stats=stats,
        )


def teacher_tab_attendance_records():
    st.header("Attendance Records")

    teacher_id = st.session_state.teacher_data[
        "teacher_id"
    ]

    records = get_attendance_for_teacher(
        teacher_id
    )

    if not records:
        return

    data = []

    for r in records:
        ts = r.get("timestamp")

        india_time = None

        if ts:
            try:
                utc_time = (
                    datetime
                    .fromisoformat(ts)
                )

                india_time = (
                    utc_time.astimezone(
                        ZoneInfo(
                            "Asia/Kolkata"
                        )
                    )
                )

            except ValueError:
                india_time = (
                    datetime.strptime(
                        ts,
                        "%d-%m-%Y %I:%M %p",
                    )
                )

        data.append(
            {
                "ts_group":
                (
                    india_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if india_time
                    else None
                ),
                "Time":
                (
                    india_time.strftime(
                        "%d-%m-%Y %I:%M %p"
                    )
                    if india_time
                    else "N/A"
                ),
                "Subject":
                r["subjects"]["name"],
                "Subject Code":
                r["subjects"][
                    "subject_code"
                ],
                "is_present":
                bool(
                    r.get(
                        "is_present",
                        False,
                    )
                ),
            }
        )

    df = pd.DataFrame(data)

    summary = (
        df.groupby(
            [
                "ts_group",
                "Time",
                "Subject",
                "Subject Code",
            ]
        )
        .agg(
            Present_Count=(
                "is_present",
                "sum",
            ),
            Total_Count=(
                "is_present",
                "count",
            ),
        )
        .reset_index()
    )

    summary[
        "Attendance Stats"
    ] = (
        "✅ "
        + summary[
            "Present_Count"
        ].astype(str)
        + " / "
        + summary[
            "Total_Count"
        ].astype(str)
        + " Students"
    )

    display_df = (
        summary.sort_values(
            by="ts_group",
            ascending=False,
        )[
            [
                "Time",
                "Subject",
                "Subject Code",
                "Attendance Stats",
            ]
        ]
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )
